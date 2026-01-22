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
class IteratorNode(ScopedExprNode):
    type = py_object_type
    iter_func_ptr = None
    counter_cname = None
    reversed = False
    is_async = False
    has_local_scope = False
    subexprs = ['sequence']

    def analyse_types(self, env):
        if self.expr_scope:
            env = self.expr_scope
        self.sequence = self.sequence.analyse_types(env)
        if (self.sequence.type.is_array or self.sequence.type.is_ptr) and (not self.sequence.type.is_string):
            self.type = self.sequence.type
        elif self.sequence.type.is_cpp_class:
            return CppIteratorNode(self.pos, sequence=self.sequence).analyse_types(env)
        elif self.is_reversed_cpp_iteration():
            sequence = self.sequence.arg_tuple.args[0].arg
            return CppIteratorNode(self.pos, sequence=sequence, reversed=True).analyse_types(env)
        else:
            self.sequence = self.sequence.coerce_to_pyobject(env)
            if self.sequence.type in (list_type, tuple_type):
                self.sequence = self.sequence.as_none_safe_node("'NoneType' object is not iterable")
        self.is_temp = 1
        return self
    gil_message = 'Iterating over Python object'
    _func_iternext_type = PyrexTypes.CPtrType(PyrexTypes.CFuncType(PyrexTypes.py_object_type, [PyrexTypes.CFuncTypeArg('it', PyrexTypes.py_object_type, None)]))

    def is_reversed_cpp_iteration(self):
        """
        Returns True if the 'reversed' function is applied to a C++ iterable.

        This supports C++ classes with reverse_iterator implemented.
        """
        if not (isinstance(self.sequence, SimpleCallNode) and self.sequence.arg_tuple and (len(self.sequence.arg_tuple.args) == 1)):
            return False
        func = self.sequence.function
        if func.is_name and func.name == 'reversed':
            if not func.entry.is_builtin:
                return False
            arg = self.sequence.arg_tuple.args[0]
            if isinstance(arg, CoercionNode) and arg.arg.is_name:
                arg = arg.arg.entry
                return arg.type.is_cpp_class
        return False

    def type_dependencies(self, env):
        return self.sequence.type_dependencies(self.expr_scope or env)

    def infer_type(self, env):
        sequence_type = self.sequence.infer_type(env)
        if sequence_type.is_array or sequence_type.is_ptr:
            return sequence_type
        elif sequence_type.is_cpp_class:
            begin = sequence_type.scope.lookup('begin')
            if begin is not None:
                return begin.type.return_type
        elif sequence_type.is_pyobject:
            return sequence_type
        return py_object_type

    def generate_result_code(self, code):
        sequence_type = self.sequence.type
        if sequence_type.is_cpp_class:
            assert False, 'Should have been changed to CppIteratorNode'
        if sequence_type.is_array or sequence_type.is_ptr:
            raise InternalError('for in carray slice not transformed')
        is_builtin_sequence = sequence_type in (list_type, tuple_type)
        if not is_builtin_sequence:
            assert not self.reversed, 'internal error: reversed() only implemented for list/tuple objects'
        self.may_be_a_sequence = not sequence_type.is_builtin_type
        if self.may_be_a_sequence:
            code.putln('if (likely(PyList_CheckExact(%s)) || PyTuple_CheckExact(%s)) {' % (self.sequence.py_result(), self.sequence.py_result()))
        if is_builtin_sequence or self.may_be_a_sequence:
            code.putln('%s = %s; __Pyx_INCREF(%s);' % (self.result(), self.sequence.py_result(), self.result()))
            self.counter_cname = code.funcstate.allocate_temp(PyrexTypes.c_py_ssize_t_type, manage_ref=False)
            if self.reversed:
                if sequence_type is list_type:
                    len_func = '__Pyx_PyList_GET_SIZE'
                else:
                    len_func = '__Pyx_PyTuple_GET_SIZE'
                code.putln('%s = %s(%s);' % (self.counter_cname, len_func, self.result()))
                code.putln('#if !CYTHON_ASSUME_SAFE_MACROS')
                code.putln(code.error_goto_if_neg(self.counter_cname, self.pos))
                code.putln('#endif')
                code.putln('--%s;' % self.counter_cname)
            else:
                code.putln('%s = 0;' % self.counter_cname)
        if not is_builtin_sequence:
            self.iter_func_ptr = code.funcstate.allocate_temp(self._func_iternext_type, manage_ref=False)
            if self.may_be_a_sequence:
                code.putln('%s = NULL;' % self.iter_func_ptr)
                code.putln('} else {')
                code.put('%s = -1; ' % self.counter_cname)
            code.putln('%s = PyObject_GetIter(%s); %s' % (self.result(), self.sequence.py_result(), code.error_goto_if_null(self.result(), self.pos)))
            self.generate_gotref(code)
            code.putln('%s = __Pyx_PyObject_GetIterNextFunc(%s); %s' % (self.iter_func_ptr, self.py_result(), code.error_goto_if_null(self.iter_func_ptr, self.pos)))
        if self.may_be_a_sequence:
            code.putln('}')

    def generate_next_sequence_item(self, test_name, result_name, code):
        assert self.counter_cname, 'internal error: counter_cname temp not prepared'
        assert test_name in ('List', 'Tuple')
        final_size = '__Pyx_Py%s_GET_SIZE(%s)' % (test_name, self.py_result())
        size_is_safe = False
        if self.sequence.is_sequence_constructor:
            item_count = len(self.sequence.args)
            if self.sequence.mult_factor is None:
                final_size = item_count
                size_is_safe = True
            elif isinstance(self.sequence.mult_factor.constant_result, _py_int_types):
                final_size = item_count * self.sequence.mult_factor.constant_result
                size_is_safe = True
        if size_is_safe:
            code.putln('if (%s >= %s) break;' % (self.counter_cname, final_size))
        else:
            code.putln('{')
            code.putln('Py_ssize_t %s = %s;' % (Naming.quick_temp_cname, final_size))
            code.putln('#if !CYTHON_ASSUME_SAFE_MACROS')
            code.putln(code.error_goto_if_neg(Naming.quick_temp_cname, self.pos))
            code.putln('#endif')
            code.putln('if (%s >= %s) break;' % (self.counter_cname, Naming.quick_temp_cname))
            code.putln('}')
        if self.reversed:
            inc_dec = '--'
        else:
            inc_dec = '++'
        code.putln('#if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS')
        code.putln('%s = Py%s_GET_ITEM(%s, %s); __Pyx_INCREF(%s); %s%s; %s' % (result_name, test_name, self.py_result(), self.counter_cname, result_name, self.counter_cname, inc_dec, code.error_goto_if_neg('0', self.pos)))
        code.putln('#else')
        code.putln('%s = __Pyx_PySequence_ITEM(%s, %s); %s%s; %s' % (result_name, self.py_result(), self.counter_cname, self.counter_cname, inc_dec, code.error_goto_if_null(result_name, self.pos)))
        code.put_gotref(result_name, py_object_type)
        code.putln('#endif')

    def generate_iter_next_result_code(self, result_name, code):
        sequence_type = self.sequence.type
        if self.reversed:
            code.putln('if (%s < 0) break;' % self.counter_cname)
        if sequence_type is list_type:
            self.generate_next_sequence_item('List', result_name, code)
            return
        elif sequence_type is tuple_type:
            self.generate_next_sequence_item('Tuple', result_name, code)
            return
        if self.may_be_a_sequence:
            code.putln('if (likely(!%s)) {' % self.iter_func_ptr)
            code.putln('if (likely(PyList_CheckExact(%s))) {' % self.py_result())
            self.generate_next_sequence_item('List', result_name, code)
            code.putln('} else {')
            self.generate_next_sequence_item('Tuple', result_name, code)
            code.putln('}')
            code.put('} else ')
        code.putln('{')
        code.putln('%s = %s(%s);' % (result_name, self.iter_func_ptr, self.py_result()))
        code.putln('if (unlikely(!%s)) {' % result_name)
        code.putln('PyObject* exc_type = PyErr_Occurred();')
        code.putln('if (exc_type) {')
        code.putln('if (likely(__Pyx_PyErr_GivenExceptionMatches(exc_type, PyExc_StopIteration))) PyErr_Clear();')
        code.putln('else %s' % code.error_goto(self.pos))
        code.putln('}')
        code.putln('break;')
        code.putln('}')
        code.put_gotref(result_name, py_object_type)
        code.putln('}')

    def free_temps(self, code):
        if self.counter_cname:
            code.funcstate.release_temp(self.counter_cname)
        if self.iter_func_ptr:
            code.funcstate.release_temp(self.iter_func_ptr)
            self.iter_func_ptr = None
        ExprNode.free_temps(self, code)