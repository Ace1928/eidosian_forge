import json
import os
import re
import shutil
import stat
import tempfile
import types
import weakref
from mako import cache
from mako import codegen
from mako import compat
from mako import exceptions
from mako import runtime
from mako import util
from mako.lexer import Lexer
class ModuleInfo:
    """Stores information about a module currently loaded into
    memory, provides reverse lookups of template source, module
    source code based on a module's identifier.

    """
    _modules = weakref.WeakValueDictionary()

    def __init__(self, module, module_filename, template, template_filename, module_source, template_source, template_uri):
        self.module = module
        self.module_filename = module_filename
        self.template_filename = template_filename
        self.module_source = module_source
        self.template_source = template_source
        self.template_uri = template_uri
        self._modules[module.__name__] = template._mmarker = self
        if module_filename:
            self._modules[module_filename] = self

    @classmethod
    def get_module_source_metadata(cls, module_source, full_line_map=False):
        source_map = re.search('__M_BEGIN_METADATA(.+?)__M_END_METADATA', module_source, re.S).group(1)
        source_map = json.loads(source_map)
        source_map['line_map'] = {int(k): int(v) for k, v in source_map['line_map'].items()}
        if full_line_map:
            f_line_map = source_map['full_line_map'] = []
            line_map = source_map['line_map']
            curr_templ_line = 1
            for mod_line in range(1, max(line_map)):
                if mod_line in line_map:
                    curr_templ_line = line_map[mod_line]
                f_line_map.append(curr_templ_line)
        return source_map

    @property
    def code(self):
        if self.module_source is not None:
            return self.module_source
        else:
            return util.read_python_file(self.module_filename)

    @property
    def source(self):
        if self.template_source is None:
            data = util.read_file(self.template_filename)
            if self.module._source_encoding:
                return data.decode(self.module._source_encoding)
            else:
                return data
        elif self.module._source_encoding and (not isinstance(self.template_source, str)):
            return self.template_source.decode(self.module._source_encoding)
        else:
            return self.template_source