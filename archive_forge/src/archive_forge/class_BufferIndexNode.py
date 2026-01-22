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
class BufferIndexNode(_IndexingBaseNode):
    """
    Indexing of buffers and memoryviews. This node is created during type
    analysis from IndexNode and replaces it.

    Attributes:
        base - base node being indexed
        indices - list of indexing expressions
    """
    subexprs = ['base', 'indices']
    is_buffer_access = True
    writable_needed = False
    index_temps = ()

    def analyse_target_types(self, env):
        self.analyse_types(env, getting=False)

    def analyse_types(self, env, getting=True):
        """
        Analyse types for buffer indexing only. Overridden by memoryview
        indexing and slicing subclasses
        """
        if not self.base.is_name and (not is_pythran_expr(self.base.type)):
            error(self.pos, 'Can only index buffer variables')
            self.type = error_type
            return self
        if not getting:
            if not self.base.entry.type.writable:
                error(self.pos, 'Writing to readonly buffer')
            else:
                self.writable_needed = True
                if self.base.type.is_buffer:
                    self.base.entry.buffer_aux.writable_needed = True
        self.none_error_message = "'NoneType' object is not subscriptable"
        self.analyse_buffer_index(env, getting)
        self.wrap_in_nonecheck_node(env)
        return self

    def analyse_buffer_index(self, env, getting):
        if is_pythran_expr(self.base.type):
            index_with_type_list = [(idx, idx.type) for idx in self.indices]
            self.type = PythranExpr(pythran_indexing_type(self.base.type, index_with_type_list))
        else:
            self.base = self.base.coerce_to_simple(env)
            self.type = self.base.type.dtype
        self.buffer_type = self.base.type
        if getting and (self.type.is_pyobject or self.type.is_pythran_expr):
            self.is_temp = True

    def analyse_assignment(self, rhs):
        """
        Called by IndexNode when this node is assigned to,
        with the rhs of the assignment
        """

    def wrap_in_nonecheck_node(self, env):
        if not env.directives['nonecheck'] or not self.base.may_be_none():
            return
        self.base = self.base.as_none_safe_node(self.none_error_message)

    def nogil_check(self, env):
        if self.is_buffer_access or self.is_memview_index:
            if self.type.is_pyobject:
                error(self.pos, 'Cannot access buffer with object dtype without gil')
                self.type = error_type

    def calculate_result_code(self):
        return '(*%s)' % self.buffer_ptr_code

    def buffer_entry(self):
        base = self.base
        if self.base.is_nonecheck:
            base = base.arg
        return base.type.get_entry(base)

    def get_index_in_temp(self, code, ivar):
        ret = code.funcstate.allocate_temp(PyrexTypes.widest_numeric_type(ivar.type, PyrexTypes.c_ssize_t_type if ivar.type.signed else PyrexTypes.c_size_t_type), manage_ref=False)
        code.putln('%s = %s;' % (ret, ivar.result()))
        return ret

    def buffer_lookup_code(self, code):
        """
        ndarray[1, 2, 3] and memslice[1, 2, 3]
        """
        if self.in_nogil_context:
            if self.is_buffer_access or self.is_memview_index:
                if code.globalstate.directives['boundscheck']:
                    warning(self.pos, 'Use boundscheck(False) for faster access', level=1)
        self.index_temps = index_temps = [self.get_index_in_temp(code, ivar) for ivar in self.indices]
        from . import Buffer
        buffer_entry = self.buffer_entry()
        if buffer_entry.type.is_buffer:
            negative_indices = buffer_entry.type.negative_indices
        else:
            negative_indices = Buffer.buffer_defaults['negative_indices']
        return (buffer_entry, Buffer.put_buffer_lookup_code(entry=buffer_entry, index_signeds=[ivar.type.signed for ivar in self.indices], index_cnames=index_temps, directives=code.globalstate.directives, pos=self.pos, code=code, negative_indices=negative_indices, in_nogil_context=self.in_nogil_context))

    def generate_assignment_code(self, rhs, code, overloaded_assignment=False):
        self.generate_subexpr_evaluation_code(code)
        self.generate_buffer_setitem_code(rhs, code)
        self.generate_subexpr_disposal_code(code)
        self.free_subexpr_temps(code)
        rhs.generate_disposal_code(code)
        rhs.free_temps(code)

    def generate_buffer_setitem_code(self, rhs, code, op=''):
        base_type = self.base.type
        if is_pythran_expr(base_type) and is_pythran_supported_type(rhs.type):
            obj = code.funcstate.allocate_temp(PythranExpr(pythran_type(self.base.type)), manage_ref=False)
            code.putln('__Pyx_call_destructor(%s);' % obj)
            code.putln('new (&%s) decltype(%s){%s};' % (obj, obj, self.base.pythran_result()))
            code.putln('%s%s %s= %s;' % (obj, pythran_indexing_code(self.indices), op, rhs.pythran_result()))
            code.funcstate.release_temp(obj)
            return
        buffer_entry, ptrexpr = self.buffer_lookup_code(code)
        if self.buffer_type.dtype.is_pyobject:
            ptr = code.funcstate.allocate_temp(buffer_entry.buf_ptr_type, manage_ref=False)
            rhs_code = rhs.result()
            code.putln('%s = %s;' % (ptr, ptrexpr))
            code.put_xgotref('*%s' % ptr, self.buffer_type.dtype)
            code.putln('__Pyx_INCREF(%s); __Pyx_XDECREF(*%s);' % (rhs_code, ptr))
            code.putln('*%s %s= %s;' % (ptr, op, rhs_code))
            code.put_xgiveref('*%s' % ptr, self.buffer_type.dtype)
            code.funcstate.release_temp(ptr)
        else:
            code.putln('*%s %s= %s;' % (ptrexpr, op, rhs.result()))

    def generate_result_code(self, code):
        if is_pythran_expr(self.base.type):
            res = self.result()
            code.putln('__Pyx_call_destructor(%s);' % res)
            code.putln('new (&%s) decltype(%s){%s%s};' % (res, res, self.base.pythran_result(), pythran_indexing_code(self.indices)))
            return
        buffer_entry, self.buffer_ptr_code = self.buffer_lookup_code(code)
        if self.type.is_pyobject:
            res = self.result()
            code.putln('%s = (PyObject *) *%s;' % (res, self.buffer_ptr_code))
            code.putln('if (unlikely(%s == NULL)) %s = Py_None;' % (res, res))
            code.putln('__Pyx_INCREF((PyObject*)%s);' % res)

    def free_subexpr_temps(self, code):
        for temp in self.index_temps:
            code.funcstate.release_temp(temp)
        self.index_temps = ()
        super(BufferIndexNode, self).free_subexpr_temps(code)