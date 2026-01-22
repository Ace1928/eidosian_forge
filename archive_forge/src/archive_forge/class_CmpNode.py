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
class CmpNode(object):
    special_bool_cmp_function = None
    special_bool_cmp_utility_code = None
    special_bool_extra_args = []

    def infer_type(self, env):
        return py_object_type

    def calculate_cascaded_constant_result(self, operand1_result):
        func = compile_time_binary_operators[self.operator]
        operand2_result = self.operand2.constant_result
        if isinstance(operand1_result, any_string_type) and isinstance(operand2_result, any_string_type) and (type(operand1_result) != type(operand2_result)):
            return
        if self.operator in ('in', 'not_in'):
            if isinstance(self.operand2, (ListNode, TupleNode, SetNode)):
                if not self.operand2.args:
                    self.constant_result = self.operator == 'not_in'
                    return
                elif isinstance(self.operand2, ListNode) and (not self.cascade):
                    self.operand2 = self.operand2.as_tuple()
            elif isinstance(self.operand2, DictNode):
                if not self.operand2.key_value_pairs:
                    self.constant_result = self.operator == 'not_in'
                    return
        self.constant_result = func(operand1_result, operand2_result)

    def cascaded_compile_time_value(self, operand1, denv):
        func = get_compile_time_binop(self)
        operand2 = self.operand2.compile_time_value(denv)
        try:
            result = func(operand1, operand2)
        except Exception as e:
            self.compile_time_value_error(e)
            result = None
        if result:
            cascade = self.cascade
            if cascade:
                result = result and cascade.cascaded_compile_time_value(operand2, denv)
        return result

    def is_cpp_comparison(self):
        return self.operand1.type.is_cpp_class or self.operand2.type.is_cpp_class

    def find_common_int_type(self, env, op, operand1, operand2):
        type1 = operand1.type
        type2 = operand2.type
        type1_can_be_int = False
        type2_can_be_int = False
        if operand1.is_string_literal and operand1.can_coerce_to_char_literal():
            type1_can_be_int = True
        if operand2.is_string_literal and operand2.can_coerce_to_char_literal():
            type2_can_be_int = True
        if type1.is_int:
            if type2_can_be_int:
                return type1
        elif type2.is_int:
            if type1_can_be_int:
                return type2
        elif type1_can_be_int:
            if type2_can_be_int:
                if Builtin.unicode_type in (type1, type2):
                    return PyrexTypes.c_py_ucs4_type
                else:
                    return PyrexTypes.c_uchar_type
        return None

    def find_common_type(self, env, op, operand1, common_type=None):
        operand2 = self.operand2
        type1 = operand1.type
        type2 = operand2.type
        new_common_type = None
        if type1 == str_type and (type2.is_string or type2 in (bytes_type, unicode_type)) or (type2 == str_type and (type1.is_string or type1 in (bytes_type, unicode_type))):
            error(self.pos, 'Comparisons between bytes/unicode and str are not portable to Python 3')
            new_common_type = error_type
        elif type1.is_complex or type2.is_complex:
            if op not in ('==', '!=') and (type1.is_complex or type1.is_numeric) and (type2.is_complex or type2.is_numeric):
                error(self.pos, 'complex types are unordered')
                new_common_type = error_type
            elif type1.is_pyobject:
                new_common_type = Builtin.complex_type if type1.subtype_of(Builtin.complex_type) else py_object_type
            elif type2.is_pyobject:
                new_common_type = Builtin.complex_type if type2.subtype_of(Builtin.complex_type) else py_object_type
            else:
                new_common_type = PyrexTypes.widest_numeric_type(type1, type2)
        elif type1.is_numeric and type2.is_numeric:
            new_common_type = PyrexTypes.widest_numeric_type(type1, type2)
        elif common_type is None or not common_type.is_pyobject:
            new_common_type = self.find_common_int_type(env, op, operand1, operand2)
        if new_common_type is None:
            if type1.is_ctuple or type2.is_ctuple:
                new_common_type = py_object_type
            elif type1 == type2:
                new_common_type = type1
            elif type1.is_pyobject or type2.is_pyobject:
                if type2.is_numeric or type2.is_string:
                    if operand2.check_for_coercion_error(type1, env):
                        new_common_type = error_type
                    else:
                        new_common_type = py_object_type
                elif type1.is_numeric or type1.is_string:
                    if operand1.check_for_coercion_error(type2, env):
                        new_common_type = error_type
                    else:
                        new_common_type = py_object_type
                elif py_object_type.assignable_from(type1) and py_object_type.assignable_from(type2):
                    new_common_type = py_object_type
                else:
                    self.invalid_types_error(operand1, op, operand2)
                    new_common_type = error_type
            elif type1.assignable_from(type2):
                new_common_type = type1
            elif type2.assignable_from(type1):
                new_common_type = type2
            else:
                self.invalid_types_error(operand1, op, operand2)
                new_common_type = error_type
        if new_common_type.is_string and (isinstance(operand1, BytesNode) or isinstance(operand2, BytesNode)):
            new_common_type = bytes_type
        if common_type is None or new_common_type.is_error:
            common_type = new_common_type
        else:
            common_type = PyrexTypes.spanning_type(common_type, new_common_type)
        if self.cascade:
            common_type = self.cascade.find_common_type(env, self.operator, operand2, common_type)
        return common_type

    def invalid_types_error(self, operand1, op, operand2):
        error(self.pos, "Invalid types for '%s' (%s, %s)" % (op, operand1.type, operand2.type))

    def is_python_comparison(self):
        return not self.is_ptr_contains() and (not self.is_c_string_contains()) and (self.has_python_operands() or (self.cascade and self.cascade.is_python_comparison()) or self.operator in ('in', 'not_in'))

    def coerce_operands_to(self, dst_type, env):
        operand2 = self.operand2
        if operand2.type != dst_type:
            self.operand2 = operand2.coerce_to(dst_type, env)
        if self.cascade:
            self.cascade.coerce_operands_to(dst_type, env)

    def is_python_result(self):
        return self.has_python_operands() and self.special_bool_cmp_function is None and (self.operator not in ('is', 'is_not', 'in', 'not_in')) and (not self.is_c_string_contains()) and (not self.is_ptr_contains()) or (self.cascade and self.cascade.is_python_result())

    def is_c_string_contains(self):
        return self.operator in ('in', 'not_in') and (self.operand1.type.is_int and (self.operand2.type.is_string or self.operand2.type is bytes_type) or (self.operand1.type.is_unicode_char and self.operand2.type is unicode_type))

    def is_ptr_contains(self):
        if self.operator in ('in', 'not_in'):
            container_type = self.operand2.type
            return (container_type.is_ptr or container_type.is_array) and (not container_type.is_string)

    def find_special_bool_compare_function(self, env, operand1, result_is_bool=False):
        if self.operator in ('==', '!='):
            type1, type2 = (operand1.type, self.operand2.type)
            if result_is_bool or (type1.is_builtin_type and type2.is_builtin_type):
                if type1 is Builtin.unicode_type or type2 is Builtin.unicode_type:
                    self.special_bool_cmp_utility_code = UtilityCode.load_cached('UnicodeEquals', 'StringTools.c')
                    self.special_bool_cmp_function = '__Pyx_PyUnicode_Equals'
                    return True
                elif type1 is Builtin.bytes_type or type2 is Builtin.bytes_type:
                    self.special_bool_cmp_utility_code = UtilityCode.load_cached('BytesEquals', 'StringTools.c')
                    self.special_bool_cmp_function = '__Pyx_PyBytes_Equals'
                    return True
                elif type1 is Builtin.basestring_type or type2 is Builtin.basestring_type:
                    self.special_bool_cmp_utility_code = UtilityCode.load_cached('UnicodeEquals', 'StringTools.c')
                    self.special_bool_cmp_function = '__Pyx_PyUnicode_Equals'
                    return True
                elif type1 is Builtin.str_type or type2 is Builtin.str_type:
                    self.special_bool_cmp_utility_code = UtilityCode.load_cached('StrEquals', 'StringTools.c')
                    self.special_bool_cmp_function = '__Pyx_PyString_Equals'
                    return True
                elif result_is_bool:
                    from .Optimize import optimise_numeric_binop
                    result = optimise_numeric_binop('Eq' if self.operator == '==' else 'Ne', self, PyrexTypes.c_bint_type, operand1, self.operand2)
                    if result:
                        self.special_bool_cmp_function, self.special_bool_cmp_utility_code, self.special_bool_extra_args, _ = result
                        return True
        elif self.operator in ('in', 'not_in'):
            if self.operand2.type is Builtin.dict_type:
                self.operand2 = self.operand2.as_none_safe_node("'NoneType' object is not iterable")
                self.special_bool_cmp_utility_code = UtilityCode.load_cached('PyDictContains', 'ObjectHandling.c')
                self.special_bool_cmp_function = '__Pyx_PyDict_ContainsTF'
                return True
            elif self.operand2.type is Builtin.set_type:
                self.operand2 = self.operand2.as_none_safe_node("'NoneType' object is not iterable")
                self.special_bool_cmp_utility_code = UtilityCode.load_cached('PySetContains', 'ObjectHandling.c')
                self.special_bool_cmp_function = '__Pyx_PySet_ContainsTF'
                return True
            elif self.operand2.type is Builtin.unicode_type:
                self.operand2 = self.operand2.as_none_safe_node("'NoneType' object is not iterable")
                self.special_bool_cmp_utility_code = UtilityCode.load_cached('PyUnicodeContains', 'StringTools.c')
                self.special_bool_cmp_function = '__Pyx_PyUnicode_ContainsTF'
                return True
            else:
                if not self.operand2.type.is_pyobject:
                    self.operand2 = self.operand2.coerce_to_pyobject(env)
                self.special_bool_cmp_utility_code = UtilityCode.load_cached('PySequenceContains', 'ObjectHandling.c')
                self.special_bool_cmp_function = '__Pyx_PySequence_ContainsTF'
                return True
        return False

    def generate_operation_code(self, code, result_code, operand1, op, operand2):
        if self.type.is_pyobject:
            error_clause = code.error_goto_if_null
            got_ref = '__Pyx_XGOTREF(%s); ' % result_code
            if self.special_bool_cmp_function:
                code.globalstate.use_utility_code(UtilityCode.load_cached('PyBoolOrNullFromLong', 'ObjectHandling.c'))
                coerce_result = '__Pyx_PyBoolOrNull_FromLong'
            else:
                coerce_result = '__Pyx_PyBool_FromLong'
        else:
            error_clause = code.error_goto_if_neg
            got_ref = ''
            coerce_result = ''
        if self.special_bool_cmp_function:
            if operand1.type.is_pyobject:
                result1 = operand1.py_result()
            else:
                result1 = operand1.result()
            if operand2.type.is_pyobject:
                result2 = operand2.py_result()
            else:
                result2 = operand2.result()
            special_bool_extra_args_result = ', '.join([extra_arg.result() for extra_arg in self.special_bool_extra_args])
            if self.special_bool_cmp_utility_code:
                code.globalstate.use_utility_code(self.special_bool_cmp_utility_code)
            code.putln('%s = %s(%s(%s, %s, %s)); %s%s' % (result_code, coerce_result, self.special_bool_cmp_function, result1, result2, special_bool_extra_args_result if self.special_bool_extra_args else richcmp_constants[op], got_ref, error_clause(result_code, self.pos)))
        elif operand1.type.is_pyobject and op not in ('is', 'is_not'):
            assert op not in ('in', 'not_in'), op
            assert self.type.is_pyobject or self.type is PyrexTypes.c_bint_type
            code.putln('%s = PyObject_RichCompare%s(%s, %s, %s); %s%s' % (result_code, '' if self.type.is_pyobject else 'Bool', operand1.py_result(), operand2.py_result(), richcmp_constants[op], got_ref, error_clause(result_code, self.pos)))
        elif operand1.type.is_complex:
            code.putln('%s = %s(%s%s(%s, %s));' % (result_code, coerce_result, op == '!=' and '!' or '', operand1.type.unary_op('eq'), operand1.result(), operand2.result()))
        else:
            type1 = operand1.type
            type2 = operand2.type
            if (type1.is_extension_type or type2.is_extension_type) and (not type1.same_as(type2)):
                common_type = py_object_type
            elif type1.is_numeric:
                common_type = PyrexTypes.widest_numeric_type(type1, type2)
            else:
                common_type = type1
            code1 = operand1.result_as(common_type)
            code2 = operand2.result_as(common_type)
            statement = '%s = %s(%s %s %s);' % (result_code, coerce_result, code1, self.c_operator(op), code2)
            if self.is_cpp_comparison() and self.exception_check == '+':
                translate_cpp_exception(code, self.pos, statement, result_code if self.type.is_pyobject else None, self.exception_value, self.in_nogil_context)
            else:
                code.putln(statement)

    def c_operator(self, op):
        if op == 'is':
            return '=='
        elif op == 'is_not':
            return '!='
        else:
            return op