from __future__ import absolute_import
import cython
import re
import sys
import copy
import os.path
import operator
from .Errors import (
from .Code import UtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from . import Nodes
from .Nodes import Node, utility_code_for_imports, SingleAssignmentNode
from . import PyrexTypes
from .PyrexTypes import py_object_type, typecast, error_type, \
from . import TypeSlots
from .Builtin import (
from . import Builtin
from . import Symtab
from .. import Utils
from .Annotate import AnnotationItem
from . import Future
from ..Debugging import print_call_chain
from .DebugFlags import debug_disposal_code, debug_coercion
from .Pythran import (to_pythran, is_pythran_supported_type, is_pythran_supported_operation_type,
from .PyrexTypes import PythranExpr
class DictNode(ExprNode):
    subexprs = ['key_value_pairs']
    is_temp = 1
    exclude_null_values = False
    type = dict_type
    is_dict_literal = True
    reject_duplicates = False
    obj_conversion_errors = []

    @classmethod
    def from_pairs(cls, pos, pairs):
        return cls(pos, key_value_pairs=[DictItemNode(pos, key=k, value=v) for k, v in pairs])

    def calculate_constant_result(self):
        self.constant_result = dict([item.constant_result for item in self.key_value_pairs])

    def compile_time_value(self, denv):
        pairs = [(item.key.compile_time_value(denv), item.value.compile_time_value(denv)) for item in self.key_value_pairs]
        try:
            return dict(pairs)
        except Exception as e:
            self.compile_time_value_error(e)

    def type_dependencies(self, env):
        return ()

    def infer_type(self, env):
        return dict_type

    def analyse_types(self, env):
        with local_errors(ignore=True) as errors:
            self.key_value_pairs = [item.analyse_types(env) for item in self.key_value_pairs]
        self.obj_conversion_errors = errors
        return self

    def may_be_none(self):
        return False

    def coerce_to(self, dst_type, env):
        if dst_type.is_pyobject:
            self.release_errors()
            if self.type.is_struct_or_union:
                if not dict_type.subtype_of(dst_type):
                    error(self.pos, "Cannot interpret struct as non-dict type '%s'" % dst_type)
                return DictNode(self.pos, key_value_pairs=[DictItemNode(item.pos, key=item.key.coerce_to_pyobject(env), value=item.value.coerce_to_pyobject(env)) for item in self.key_value_pairs])
            if not self.type.subtype_of(dst_type):
                error(self.pos, "Cannot interpret dict as type '%s'" % dst_type)
        elif dst_type.is_struct_or_union:
            self.type = dst_type
            if not dst_type.is_struct and len(self.key_value_pairs) != 1:
                error(self.pos, "Exactly one field must be specified to convert to union '%s'" % dst_type)
            elif dst_type.is_struct and len(self.key_value_pairs) < len(dst_type.scope.var_entries):
                warning(self.pos, "Not all members given for struct '%s'" % dst_type, 1)
            for item in self.key_value_pairs:
                if isinstance(item.key, CoerceToPyTypeNode):
                    item.key = item.key.arg
                if not item.key.is_string_literal:
                    error(item.key.pos, 'Invalid struct field identifier')
                    item.key = StringNode(item.key.pos, value='<error>')
                else:
                    key = str(item.key.value)
                    member = dst_type.scope.lookup_here(key)
                    if not member:
                        error(item.key.pos, "struct '%s' has no field '%s'" % (dst_type, key))
                    else:
                        value = item.value
                        if isinstance(value, CoerceToPyTypeNode):
                            value = value.arg
                        item.value = value.coerce_to(member.type, env)
        else:
            return super(DictNode, self).coerce_to(dst_type, env)
        return self

    def release_errors(self):
        for err in self.obj_conversion_errors:
            report_error(err)
        self.obj_conversion_errors = []
    gil_message = 'Constructing Python dict'

    def generate_evaluation_code(self, code):
        code.mark_pos(self.pos)
        self.allocate_temp_result(code)
        is_dict = self.type.is_pyobject
        if is_dict:
            self.release_errors()
            code.putln('%s = __Pyx_PyDict_NewPresized(%d); %s' % (self.result(), len(self.key_value_pairs), code.error_goto_if_null(self.result(), self.pos)))
            self.generate_gotref(code)
        keys_seen = set()
        key_type = None
        needs_error_helper = False
        for item in self.key_value_pairs:
            item.generate_evaluation_code(code)
            if is_dict:
                if self.exclude_null_values:
                    code.putln('if (%s) {' % item.value.py_result())
                key = item.key
                if self.reject_duplicates:
                    if keys_seen is not None:
                        if not key.is_string_literal:
                            keys_seen = None
                        elif key.value in keys_seen:
                            keys_seen = None
                        elif key_type is not type(key.value):
                            if key_type is None:
                                key_type = type(key.value)
                                keys_seen.add(key.value)
                            else:
                                keys_seen = None
                        else:
                            keys_seen.add(key.value)
                    if keys_seen is None:
                        code.putln('if (unlikely(PyDict_Contains(%s, %s))) {' % (self.result(), key.py_result()))
                        needs_error_helper = True
                        code.putln('__Pyx_RaiseDoubleKeywordsError("function", %s); %s' % (key.py_result(), code.error_goto(item.pos)))
                        code.putln('} else {')
                code.put_error_if_neg(self.pos, 'PyDict_SetItem(%s, %s, %s)' % (self.result(), item.key.py_result(), item.value.py_result()))
                if self.reject_duplicates and keys_seen is None:
                    code.putln('}')
                if self.exclude_null_values:
                    code.putln('}')
            elif item.value.type.is_array:
                code.putln('memcpy(%s.%s, %s, sizeof(%s));' % (self.result(), item.key.value, item.value.result(), item.value.result()))
            else:
                code.putln('%s.%s = %s;' % (self.result(), item.key.value, item.value.result()))
            item.generate_disposal_code(code)
            item.free_temps(code)
        if needs_error_helper:
            code.globalstate.use_utility_code(UtilityCode.load_cached('RaiseDoubleKeywords', 'FunctionArguments.c'))

    def annotate(self, code):
        for item in self.key_value_pairs:
            item.annotate(code)

    def as_python_dict(self):
        return dict([(key.value, value) for key, value in self.key_value_pairs])