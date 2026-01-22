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
class SliceIndexNode(ExprNode):
    subexprs = ['base', 'start', 'stop', 'slice']
    nogil = False
    slice = None

    def infer_type(self, env):
        base_type = self.base.infer_type(env)
        if base_type.is_string or base_type.is_cpp_class:
            return bytes_type
        elif base_type.is_pyunicode_ptr:
            return unicode_type
        elif base_type in (bytes_type, bytearray_type, str_type, unicode_type, basestring_type, list_type, tuple_type):
            return base_type
        elif base_type.is_ptr or base_type.is_array:
            return PyrexTypes.c_array_type(base_type.base_type, None)
        return py_object_type

    def inferable_item_node(self, index=0):
        if index is not not_a_constant and self.start:
            if self.start.has_constant_result():
                index += self.start.constant_result
            else:
                index = not_a_constant
        return self.base.inferable_item_node(index)

    def may_be_none(self):
        base_type = self.base.type
        if base_type:
            if base_type.is_string:
                return False
            if base_type in (bytes_type, str_type, unicode_type, basestring_type, list_type, tuple_type):
                return False
        return ExprNode.may_be_none(self)

    def calculate_constant_result(self):
        if self.start is None:
            start = None
        else:
            start = self.start.constant_result
        if self.stop is None:
            stop = None
        else:
            stop = self.stop.constant_result
        self.constant_result = self.base.constant_result[start:stop]

    def compile_time_value(self, denv):
        base = self.base.compile_time_value(denv)
        if self.start is None:
            start = 0
        else:
            start = self.start.compile_time_value(denv)
        if self.stop is None:
            stop = None
        else:
            stop = self.stop.compile_time_value(denv)
        try:
            return base[start:stop]
        except Exception as e:
            self.compile_time_value_error(e)

    def analyse_target_declaration(self, env):
        pass

    def analyse_target_types(self, env):
        node = self.analyse_types(env, getting=False)
        if node.type.is_pyobject:
            node.type = py_object_type
        return node

    def analyse_types(self, env, getting=True):
        self.base = self.base.analyse_types(env)
        if self.base.type.is_buffer or self.base.type.is_pythran_expr or self.base.type.is_memoryviewslice:
            none_node = NoneNode(self.pos)
            index = SliceNode(self.pos, start=self.start or none_node, stop=self.stop or none_node, step=none_node)
            index_node = IndexNode(self.pos, index=index, base=self.base)
            return index_node.analyse_base_and_index_types(env, getting=getting, setting=not getting, analyse_base=False)
        if self.start:
            self.start = self.start.analyse_types(env)
        if self.stop:
            self.stop = self.stop.analyse_types(env)
        if not env.directives['wraparound']:
            check_negative_indices(self.start, self.stop)
        base_type = self.base.type
        if base_type.is_array and (not getting):
            if not self.start and (not self.stop):
                self.type = base_type
            else:
                self.type = PyrexTypes.CPtrType(base_type.base_type)
        elif base_type.is_string or base_type.is_cpp_string:
            self.type = default_str_type(env)
        elif base_type.is_pyunicode_ptr:
            self.type = unicode_type
        elif base_type.is_ptr:
            self.type = base_type
        elif base_type.is_array:
            self.type = PyrexTypes.CPtrType(base_type.base_type)
        else:
            self.base = self.base.coerce_to_pyobject(env)
            self.type = py_object_type
        if base_type.is_builtin_type:
            self.type = base_type
            self.base = self.base.as_none_safe_node("'NoneType' object is not subscriptable")
        if self.type is py_object_type:
            if (not self.start or self.start.is_literal) and (not self.stop or self.stop.is_literal):
                none_node = NoneNode(self.pos)
                self.slice = SliceNode(self.pos, start=copy.deepcopy(self.start or none_node), stop=copy.deepcopy(self.stop or none_node), step=none_node).analyse_types(env)
        else:
            c_int = PyrexTypes.c_py_ssize_t_type

            def allow_none(node, default_value, env):
                from .UtilNodes import EvalWithTempExprNode, ResultRefNode
                node_ref = ResultRefNode(node)
                new_expr = CondExprNode(node.pos, true_val=IntNode(node.pos, type=c_int, value=default_value, constant_result=int(default_value) if default_value.isdigit() else not_a_constant), false_val=node_ref.coerce_to(c_int, env), test=PrimaryCmpNode(node.pos, operand1=node_ref, operator='is', operand2=NoneNode(node.pos)).analyse_types(env)).analyse_result_type(env)
                return EvalWithTempExprNode(node_ref, new_expr)
            if self.start:
                if self.start.type.is_pyobject:
                    self.start = allow_none(self.start, '0', env)
                self.start = self.start.coerce_to(c_int, env)
            if self.stop:
                if self.stop.type.is_pyobject:
                    self.stop = allow_none(self.stop, 'PY_SSIZE_T_MAX', env)
                self.stop = self.stop.coerce_to(c_int, env)
        self.is_temp = 1
        return self

    def analyse_as_type(self, env):
        base_type = self.base.analyse_as_type(env)
        if base_type:
            if not self.start and (not self.stop):
                from . import MemoryView
                env.use_utility_code(MemoryView.view_utility_code)
                none_node = NoneNode(self.pos)
                slice_node = SliceNode(self.pos, start=none_node, stop=none_node, step=none_node)
                return PyrexTypes.MemoryViewSliceType(base_type, MemoryView.get_axes_specs(env, [slice_node]))
        return None

    def nogil_check(self, env):
        self.nogil = env.nogil
        return super(SliceIndexNode, self).nogil_check(env)
    gil_message = 'Slicing Python object'
    get_slice_utility_code = TempitaUtilityCode.load('SliceObject', 'ObjectHandling.c', context={'access': 'Get'})
    set_slice_utility_code = TempitaUtilityCode.load('SliceObject', 'ObjectHandling.c', context={'access': 'Set'})

    def coerce_to(self, dst_type, env):
        if (self.base.type.is_string or self.base.type.is_cpp_string) and dst_type in (bytes_type, bytearray_type, str_type, unicode_type):
            if dst_type not in (bytes_type, bytearray_type) and (not env.directives['c_string_encoding']):
                error(self.pos, "default encoding required for conversion from '%s' to '%s'" % (self.base.type, dst_type))
            self.type = dst_type
        if dst_type.is_array and self.base.type.is_array:
            if not self.start and (not self.stop):
                return self.base.coerce_to(dst_type, env)
        return super(SliceIndexNode, self).coerce_to(dst_type, env)

    def generate_result_code(self, code):
        if not self.type.is_pyobject:
            error(self.pos, "Slicing is not currently supported for '%s'." % self.type)
            return
        base_result = self.base.result()
        result = self.result()
        start_code = self.start_code()
        stop_code = self.stop_code()
        if self.base.type.is_string:
            base_result = self.base.result()
            if self.base.type not in (PyrexTypes.c_char_ptr_type, PyrexTypes.c_const_char_ptr_type):
                base_result = '((const char*)%s)' % base_result
            if self.type is bytearray_type:
                type_name = 'ByteArray'
            else:
                type_name = self.type.name.title()
            if self.stop is None:
                code.putln('%s = __Pyx_Py%s_FromString(%s + %s); %s' % (result, type_name, base_result, start_code, code.error_goto_if_null(result, self.pos)))
            else:
                code.putln('%s = __Pyx_Py%s_FromStringAndSize(%s + %s, %s - %s); %s' % (result, type_name, base_result, start_code, stop_code, start_code, code.error_goto_if_null(result, self.pos)))
        elif self.base.type.is_pyunicode_ptr:
            base_result = self.base.result()
            if self.base.type != PyrexTypes.c_py_unicode_ptr_type:
                base_result = '((const Py_UNICODE*)%s)' % base_result
            if self.stop is None:
                code.putln('%s = __Pyx_PyUnicode_FromUnicode(%s + %s); %s' % (result, base_result, start_code, code.error_goto_if_null(result, self.pos)))
                code.globalstate.use_utility_code(UtilityCode.load_cached('pyunicode_from_unicode', 'StringTools.c'))
            else:
                code.putln('%s = __Pyx_PyUnicode_FromUnicodeAndLength(%s + %s, %s - %s); %s' % (result, base_result, start_code, stop_code, start_code, code.error_goto_if_null(result, self.pos)))
                code.globalstate.use_utility_code(UtilityCode.load_cached('pyunicode_from_unicode', 'StringTools.c'))
        elif self.base.type is unicode_type:
            code.globalstate.use_utility_code(UtilityCode.load_cached('PyUnicode_Substring', 'StringTools.c'))
            code.putln('%s = __Pyx_PyUnicode_Substring(%s, %s, %s); %s' % (result, base_result, start_code, stop_code, code.error_goto_if_null(result, self.pos)))
        elif self.type is py_object_type:
            code.globalstate.use_utility_code(self.get_slice_utility_code)
            has_c_start, has_c_stop, c_start, c_stop, py_start, py_stop, py_slice = self.get_slice_config()
            code.putln('%s = __Pyx_PyObject_GetSlice(%s, %s, %s, %s, %s, %s, %d, %d, %d); %s' % (result, self.base.py_result(), c_start, c_stop, py_start, py_stop, py_slice, has_c_start, has_c_stop, bool(code.globalstate.directives['wraparound']), code.error_goto_if_null(result, self.pos)))
        else:
            if self.base.type is list_type:
                code.globalstate.use_utility_code(TempitaUtilityCode.load_cached('SliceTupleAndList', 'ObjectHandling.c'))
                cfunc = '__Pyx_PyList_GetSlice'
            elif self.base.type is tuple_type:
                code.globalstate.use_utility_code(TempitaUtilityCode.load_cached('SliceTupleAndList', 'ObjectHandling.c'))
                cfunc = '__Pyx_PyTuple_GetSlice'
            else:
                cfunc = 'PySequence_GetSlice'
            code.putln('%s = %s(%s, %s, %s); %s' % (result, cfunc, self.base.py_result(), start_code, stop_code, code.error_goto_if_null(result, self.pos)))
        self.generate_gotref(code)

    def generate_assignment_code(self, rhs, code, overloaded_assignment=False, exception_check=None, exception_value=None):
        self.generate_subexpr_evaluation_code(code)
        if self.type.is_pyobject:
            code.globalstate.use_utility_code(self.set_slice_utility_code)
            has_c_start, has_c_stop, c_start, c_stop, py_start, py_stop, py_slice = self.get_slice_config()
            code.put_error_if_neg(self.pos, '__Pyx_PyObject_SetSlice(%s, %s, %s, %s, %s, %s, %s, %d, %d, %d)' % (self.base.py_result(), rhs.py_result(), c_start, c_stop, py_start, py_stop, py_slice, has_c_start, has_c_stop, bool(code.globalstate.directives['wraparound'])))
        else:
            start_offset = self.start_code() if self.start else '0'
            if rhs.type.is_array:
                array_length = rhs.type.size
                self.generate_slice_guard_code(code, array_length)
            else:
                array_length = '%s - %s' % (self.stop_code(), start_offset)
            code.globalstate.use_utility_code(UtilityCode.load_cached('IncludeStringH', 'StringTools.c'))
            code.putln('memcpy(&(%s[%s]), %s, sizeof(%s[0]) * (%s));' % (self.base.result(), start_offset, rhs.result(), self.base.result(), array_length))
        self.generate_subexpr_disposal_code(code)
        self.free_subexpr_temps(code)
        rhs.generate_disposal_code(code)
        rhs.free_temps(code)

    def generate_deletion_code(self, code, ignore_nonexisting=False):
        if not self.base.type.is_pyobject:
            error(self.pos, "Deleting slices is only supported for Python types, not '%s'." % self.type)
            return
        self.generate_subexpr_evaluation_code(code)
        code.globalstate.use_utility_code(self.set_slice_utility_code)
        has_c_start, has_c_stop, c_start, c_stop, py_start, py_stop, py_slice = self.get_slice_config()
        code.put_error_if_neg(self.pos, '__Pyx_PyObject_DelSlice(%s, %s, %s, %s, %s, %s, %d, %d, %d)' % (self.base.py_result(), c_start, c_stop, py_start, py_stop, py_slice, has_c_start, has_c_stop, bool(code.globalstate.directives['wraparound'])))
        self.generate_subexpr_disposal_code(code)
        self.free_subexpr_temps(code)

    def get_slice_config(self):
        has_c_start, c_start, py_start = (False, '0', 'NULL')
        if self.start:
            has_c_start = not self.start.type.is_pyobject
            if has_c_start:
                c_start = self.start.result()
            else:
                py_start = '&%s' % self.start.py_result()
        has_c_stop, c_stop, py_stop = (False, '0', 'NULL')
        if self.stop:
            has_c_stop = not self.stop.type.is_pyobject
            if has_c_stop:
                c_stop = self.stop.result()
            else:
                py_stop = '&%s' % self.stop.py_result()
        py_slice = self.slice and '&%s' % self.slice.py_result() or 'NULL'
        return (has_c_start, has_c_stop, c_start, c_stop, py_start, py_stop, py_slice)

    def generate_slice_guard_code(self, code, target_size):
        if not self.base.type.is_array:
            return
        slice_size = self.base.type.size
        try:
            total_length = slice_size = int(slice_size)
        except ValueError:
            total_length = None
        start = stop = None
        if self.stop:
            stop = self.stop.result()
            try:
                stop = int(stop)
                if stop < 0:
                    if total_length is None:
                        slice_size = '%s + %d' % (slice_size, stop)
                    else:
                        slice_size += stop
                else:
                    slice_size = stop
                stop = None
            except ValueError:
                pass
        if self.start:
            start = self.start.result()
            try:
                start = int(start)
                if start < 0:
                    if total_length is None:
                        start = '%s + %d' % (self.base.type.size, start)
                    else:
                        start += total_length
                if isinstance(slice_size, _py_int_types):
                    slice_size -= start
                else:
                    slice_size = '%s - (%s)' % (slice_size, start)
                start = None
            except ValueError:
                pass
        runtime_check = None
        compile_time_check = False
        try:
            int_target_size = int(target_size)
        except ValueError:
            int_target_size = None
        else:
            compile_time_check = isinstance(slice_size, _py_int_types)
        if compile_time_check and slice_size < 0:
            if int_target_size > 0:
                error(self.pos, 'Assignment to empty slice.')
        elif compile_time_check and start is None and (stop is None):
            if int_target_size != slice_size:
                error(self.pos, 'Assignment to slice of wrong length, expected %s, got %s' % (slice_size, target_size))
        elif start is not None:
            if stop is None:
                stop = slice_size
            runtime_check = '(%s)-(%s)' % (stop, start)
        elif stop is not None:
            runtime_check = stop
        else:
            runtime_check = slice_size
        if runtime_check:
            code.putln('if (unlikely((%s) != (%s))) {' % (runtime_check, target_size))
            if self.nogil:
                code.put_ensure_gil()
            code.putln('PyErr_Format(PyExc_ValueError, "Assignment to slice of wrong length, expected %%" CYTHON_FORMAT_SSIZE_T "d, got %%" CYTHON_FORMAT_SSIZE_T "d", (Py_ssize_t)(%s), (Py_ssize_t)(%s));' % (target_size, runtime_check))
            if self.nogil:
                code.put_release_ensured_gil()
            code.putln(code.error_goto(self.pos))
            code.putln('}')

    def start_code(self):
        if self.start:
            return self.start.result()
        else:
            return '0'

    def stop_code(self):
        if self.stop:
            return self.stop.result()
        elif self.base.type.is_array:
            return self.base.type.size
        else:
            return 'PY_SSIZE_T_MAX'

    def calculate_result_code(self):
        return '<unused>'