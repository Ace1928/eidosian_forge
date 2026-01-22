from __future__ import absolute_import
import cython
import sys, copy
from itertools import chain
from . import Builtin
from .Errors import error, warning, InternalError, CompileError, CannotSpecialize
from . import Naming
from . import PyrexTypes
from . import TypeSlots
from .PyrexTypes import py_object_type, error_type
from .Symtab import (ModuleScope, LocalScope, ClosureScope, PropertyScope,
from .Code import UtilityCode
from .StringEncoding import EncodedString
from . import Future
from . import Options
from . import DebugFlags
from .Pythran import has_np_pythran, pythran_type, is_pythran_buffer
from ..Utils import add_metaclass, str_to_number
class DictIterationNextNode(Node):
    child_attrs = ['dict_obj', 'expected_size', 'pos_index_var', 'coerced_key_var', 'coerced_value_var', 'coerced_tuple_var', 'key_target', 'value_target', 'tuple_target', 'is_dict_flag']
    coerced_key_var = key_ref = None
    coerced_value_var = value_ref = None
    coerced_tuple_var = tuple_ref = None

    def __init__(self, dict_obj, expected_size, pos_index_var, key_target, value_target, tuple_target, is_dict_flag):
        Node.__init__(self, dict_obj.pos, dict_obj=dict_obj, expected_size=expected_size, pos_index_var=pos_index_var, key_target=key_target, value_target=value_target, tuple_target=tuple_target, is_dict_flag=is_dict_flag, is_temp=True, type=PyrexTypes.c_bint_type)

    def analyse_expressions(self, env):
        from . import ExprNodes
        self.dict_obj = self.dict_obj.analyse_types(env)
        self.expected_size = self.expected_size.analyse_types(env)
        if self.pos_index_var:
            self.pos_index_var = self.pos_index_var.analyse_types(env)
        if self.key_target:
            self.key_target = self.key_target.analyse_target_types(env)
            self.key_ref = ExprNodes.TempNode(self.key_target.pos, PyrexTypes.py_object_type)
            self.coerced_key_var = self.key_ref.coerce_to(self.key_target.type, env)
        if self.value_target:
            self.value_target = self.value_target.analyse_target_types(env)
            self.value_ref = ExprNodes.TempNode(self.value_target.pos, type=PyrexTypes.py_object_type)
            self.coerced_value_var = self.value_ref.coerce_to(self.value_target.type, env)
        if self.tuple_target:
            self.tuple_target = self.tuple_target.analyse_target_types(env)
            self.tuple_ref = ExprNodes.TempNode(self.tuple_target.pos, PyrexTypes.py_object_type)
            self.coerced_tuple_var = self.tuple_ref.coerce_to(self.tuple_target.type, env)
        self.is_dict_flag = self.is_dict_flag.analyse_types(env)
        return self

    def generate_function_definitions(self, env, code):
        self.dict_obj.generate_function_definitions(env, code)

    def generate_execution_code(self, code):
        code.globalstate.use_utility_code(UtilityCode.load_cached('dict_iter', 'Optimize.c'))
        self.dict_obj.generate_evaluation_code(code)
        assignments = []
        temp_addresses = []
        for var, result, target in [(self.key_ref, self.coerced_key_var, self.key_target), (self.value_ref, self.coerced_value_var, self.value_target), (self.tuple_ref, self.coerced_tuple_var, self.tuple_target)]:
            if target is None:
                addr = 'NULL'
            else:
                assignments.append((var, result, target))
                var.allocate(code)
                addr = '&%s' % var.result()
            temp_addresses.append(addr)
        result_temp = code.funcstate.allocate_temp(PyrexTypes.c_int_type, False)
        code.putln('%s = __Pyx_dict_iter_next(%s, %s, &%s, %s, %s, %s, %s);' % (result_temp, self.dict_obj.py_result(), self.expected_size.result(), self.pos_index_var.result(), temp_addresses[0], temp_addresses[1], temp_addresses[2], self.is_dict_flag.result()))
        code.putln('if (unlikely(%s == 0)) break;' % result_temp)
        code.putln(code.error_goto_if('%s == -1' % result_temp, self.pos))
        code.funcstate.release_temp(result_temp)
        for var, result, target in assignments:
            var.generate_gotref(code)
        for var, result, target in assignments:
            result.generate_evaluation_code(code)
        for var, result, target in assignments:
            target.generate_assignment_code(result, code)
            var.release(code)