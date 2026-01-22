from __future__ import absolute_import
import cython
import hashlib
import operator
import os
import re
import shutil
import textwrap
from string import Template
from functools import partial
from contextlib import closing, contextmanager
from collections import defaultdict
from . import Naming
from . import Options
from . import DebugFlags
from . import StringEncoding
from .. import Utils
from .Scanning import SourceDescriptor
from ..StringIOTree import StringIOTree
class UtilityCode(UtilityCodeBase):
    """
    Stores utility code to add during code generation.

    See GlobalState.put_utility_code.

    hashes/equals by instance

    proto           C prototypes
    impl            implementation code
    init            code to call on module initialization
    requires        utility code dependencies
    proto_block     the place in the resulting file where the prototype should
                    end up
    name            name of the utility code (or None)
    file            filename of the utility code file this utility was loaded
                    from (or None)
    """

    def __init__(self, proto=None, impl=None, init=None, cleanup=None, requires=None, proto_block='utility_code_proto', name=None, file=None):
        self.proto = proto
        self.impl = impl
        self.init = init
        self.cleanup = cleanup
        self.requires = requires
        self._cache = {}
        self.specialize_list = []
        self.proto_block = proto_block
        self.name = name
        self.file = file

    def __hash__(self):
        return hash((self.proto, self.impl))

    def __eq__(self, other):
        if self is other:
            return True
        self_type, other_type = (type(self), type(other))
        if self_type is not other_type and (not (isinstance(other, self_type) or isinstance(self, other_type))):
            return False
        self_init = getattr(self, 'init', None)
        other_init = getattr(other, 'init', None)
        self_proto = getattr(self, 'proto', None)
        other_proto = getattr(other, 'proto', None)
        return (self_init, self_proto, self.impl) == (other_init, other_proto, other.impl)

    def none_or_sub(self, s, context):
        """
        Format a string in this utility code with context. If None, do nothing.
        """
        if s is None:
            return None
        return s % context

    def specialize(self, pyrex_type=None, **data):
        name = self.name
        if pyrex_type is not None:
            data['type'] = pyrex_type.empty_declaration_code()
            data['type_name'] = pyrex_type.specialization_name()
            name = '%s[%s]' % (name, data['type_name'])
        key = tuple(sorted(data.items()))
        try:
            return self._cache[key]
        except KeyError:
            if self.requires is None:
                requires = None
            else:
                requires = [r.specialize(data) for r in self.requires]
            s = self._cache[key] = UtilityCode(self.none_or_sub(self.proto, data), self.none_or_sub(self.impl, data), self.none_or_sub(self.init, data), self.none_or_sub(self.cleanup, data), requires, self.proto_block, name)
            self.specialize_list.append(s)
            return s

    def inject_string_constants(self, impl, output):
        """Replace 'PYIDENT("xyz")' by a constant Python identifier cname.
        """
        if 'PYIDENT(' not in impl and 'PYUNICODE(' not in impl:
            return (False, impl)
        replacements = {}

        def externalise(matchobj):
            key = matchobj.groups()
            try:
                cname = replacements[key]
            except KeyError:
                str_type, name = key
                cname = replacements[key] = output.get_py_string_const(StringEncoding.EncodedString(name), identifier=str_type == 'IDENT').cname
            return cname
        impl = re.sub('PY(IDENT|UNICODE)\\("([^"]+)"\\)', externalise, impl)
        assert 'PYIDENT(' not in impl and 'PYUNICODE(' not in impl
        return (True, impl)

    def inject_unbound_methods(self, impl, output):
        """Replace 'UNBOUND_METHOD(type, "name")' by a constant Python identifier cname.
        """
        if 'CALL_UNBOUND_METHOD(' not in impl:
            return (False, impl)

        def externalise(matchobj):
            type_cname, method_name, obj_cname, args = matchobj.groups()
            args = [arg.strip() for arg in args[1:].split(',')] if args else []
            assert len(args) < 3, 'CALL_UNBOUND_METHOD() does not support %d call arguments' % len(args)
            return output.cached_unbound_method_call_code(obj_cname, type_cname, method_name, args)
        impl = re.sub('CALL_UNBOUND_METHOD\\(([a-zA-Z_]+),\\s*"([^"]+)",\\s*([^),]+)((?:,[^),]+)*)\\)', externalise, impl)
        assert 'CALL_UNBOUND_METHOD(' not in impl
        return (True, impl)

    def wrap_c_strings(self, impl):
        """Replace CSTRING('''xyz''') by a C compatible string
        """
        if 'CSTRING(' not in impl:
            return impl

        def split_string(matchobj):
            content = matchobj.group(1).replace('"', '"')
            return ''.join(('"%s\\n"\n' % line if not line.endswith('\\') or line.endswith('\\\\') else '"%s"\n' % line[:-1] for line in content.splitlines()))
        impl = re.sub('CSTRING\\(\\s*"""([^"]*(?:"[^"]+)*)"""\\s*\\)', split_string, impl)
        assert 'CSTRING(' not in impl
        return impl

    def put_code(self, output):
        if self.requires:
            for dependency in self.requires:
                output.use_utility_code(dependency)
        if self.proto:
            writer = output[self.proto_block]
            writer.putln('/* %s.proto */' % self.name)
            writer.put_or_include(self.format_code(self.proto), '%s_proto' % self.name)
        if self.impl:
            impl = self.format_code(self.wrap_c_strings(self.impl))
            is_specialised1, impl = self.inject_string_constants(impl, output)
            is_specialised2, impl = self.inject_unbound_methods(impl, output)
            writer = output['utility_code_def']
            writer.putln('/* %s */' % self.name)
            if not (is_specialised1 or is_specialised2):
                writer.put_or_include(impl, '%s_impl' % self.name)
            else:
                writer.put(impl)
        if self.init:
            writer = output['init_globals']
            writer.putln('/* %s.init */' % self.name)
            if isinstance(self.init, basestring):
                writer.put(self.format_code(self.init))
            else:
                self.init(writer, output.module_pos)
            writer.putln(writer.error_goto_if_PyErr(output.module_pos))
            writer.putln()
        if self.cleanup and Options.generate_cleanup_code:
            writer = output['cleanup_globals']
            writer.putln('/* %s.cleanup */' % self.name)
            if isinstance(self.cleanup, basestring):
                writer.put_or_include(self.format_code(self.cleanup), '%s_cleanup' % self.name)
            else:
                self.cleanup(writer, output.module_pos)