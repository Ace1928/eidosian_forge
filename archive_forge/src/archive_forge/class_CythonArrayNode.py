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
class CythonArrayNode(ExprNode):
    """
    Used when a pointer of base_type is cast to a memoryviewslice with that
    base type. i.e.

        <int[:M:1, :N]> p

    creates a fortran-contiguous cython.array.

    We leave the type set to object so coercions to object are more efficient
    and less work. Acquiring a memoryviewslice from this will be just as
    efficient. ExprNode.coerce_to() will do the additional typecheck on
    self.compile_time_type

    This also handles <int[:, :]> my_c_array


    operand             ExprNode                 the thing we're casting
    base_type_node      MemoryViewSliceTypeNode  the cast expression node
    """
    subexprs = ['operand', 'shapes']
    shapes = None
    is_temp = True
    mode = 'c'
    array_dtype = None
    shape_type = PyrexTypes.c_py_ssize_t_type

    def analyse_types(self, env):
        from . import MemoryView
        self.operand = self.operand.analyse_types(env)
        if self.array_dtype:
            array_dtype = self.array_dtype
        else:
            array_dtype = self.base_type_node.base_type_node.analyse(env)
        axes = self.base_type_node.axes
        self.type = error_type
        self.shapes = []
        ndim = len(axes)
        base_type = self.operand.type
        if not self.operand.type.is_ptr and (not self.operand.type.is_array):
            error(self.operand.pos, ERR_NOT_POINTER)
            return self
        array_dimension_sizes = []
        if base_type.is_array:
            while base_type.is_array:
                array_dimension_sizes.append(base_type.size)
                base_type = base_type.base_type
        elif base_type.is_ptr:
            base_type = base_type.base_type
        else:
            error(self.pos, 'unexpected base type %s found' % base_type)
            return self
        if not (base_type.same_as(array_dtype) or base_type.is_void):
            error(self.operand.pos, ERR_BASE_TYPE)
            return self
        elif self.operand.type.is_array and len(array_dimension_sizes) != ndim:
            error(self.operand.pos, 'Expected %d dimensions, array has %d dimensions' % (ndim, len(array_dimension_sizes)))
            return self
        for axis_no, axis in enumerate(axes):
            if not axis.start.is_none:
                error(axis.start.pos, ERR_START)
                return self
            if axis.stop.is_none:
                if array_dimension_sizes:
                    dimsize = array_dimension_sizes[axis_no]
                    axis.stop = IntNode(self.pos, value=str(dimsize), constant_result=dimsize, type=PyrexTypes.c_int_type)
                else:
                    error(axis.pos, ERR_NOT_STOP)
                    return self
            axis.stop = axis.stop.analyse_types(env)
            shape = axis.stop.coerce_to(self.shape_type, env)
            if not shape.is_literal:
                shape.coerce_to_temp(env)
            self.shapes.append(shape)
            first_or_last = axis_no in (0, ndim - 1)
            if not axis.step.is_none and first_or_last:
                axis.step = axis.step.analyse_types(env)
                if not axis.step.type.is_int and axis.step.is_literal and (not axis.step.type.is_error):
                    error(axis.step.pos, 'Expected an integer literal')
                    return self
                if axis.step.compile_time_value(env) != 1:
                    error(axis.step.pos, ERR_STEPS)
                    return self
                if axis_no == 0:
                    self.mode = 'fortran'
            elif not axis.step.is_none and (not first_or_last):
                error(axis.step.pos, ERR_STEPS)
                return self
        if not self.operand.is_name:
            self.operand = self.operand.coerce_to_temp(env)
        axes = [('direct', 'follow')] * len(axes)
        if self.mode == 'fortran':
            axes[0] = ('direct', 'contig')
        else:
            axes[-1] = ('direct', 'contig')
        self.coercion_type = PyrexTypes.MemoryViewSliceType(array_dtype, axes)
        self.coercion_type.validate_memslice_dtype(self.pos)
        self.type = self.get_cython_array_type(env)
        MemoryView.use_cython_array_utility_code(env)
        env.use_utility_code(MemoryView.typeinfo_to_format_code)
        return self

    def allocate_temp_result(self, code):
        if self.temp_code:
            raise RuntimeError('temp allocated multiple times')
        self.temp_code = code.funcstate.allocate_temp(self.type, True)

    def infer_type(self, env):
        return self.get_cython_array_type(env)

    def get_cython_array_type(self, env):
        cython_scope = env.global_scope().context.cython_scope
        cython_scope.load_cythonscope()
        return cython_scope.viewscope.lookup('array').type

    def generate_result_code(self, code):
        from . import Buffer
        shapes = [self.shape_type.cast_code(shape.result()) for shape in self.shapes]
        dtype = self.coercion_type.dtype
        shapes_temp = code.funcstate.allocate_temp(py_object_type, True)
        format_temp = code.funcstate.allocate_temp(py_object_type, True)
        itemsize = 'sizeof(%s)' % dtype.empty_declaration_code()
        type_info = Buffer.get_type_information_cname(code, dtype)
        if self.operand.type.is_ptr:
            code.putln('if (!%s) {' % self.operand.result())
            code.putln('PyErr_SetString(PyExc_ValueError,"Cannot create cython.array from NULL pointer");')
            code.putln(code.error_goto(self.operand.pos))
            code.putln('}')
        code.putln('%s = __pyx_format_from_typeinfo(&%s); %s' % (format_temp, type_info, code.error_goto_if_null(format_temp, self.pos)))
        code.put_gotref(format_temp, py_object_type)
        buildvalue_fmt = ' __PYX_BUILD_PY_SSIZE_T ' * len(shapes)
        code.putln('%s = Py_BuildValue((char*) "(" %s ")", %s); %s' % (shapes_temp, buildvalue_fmt, ', '.join(shapes), code.error_goto_if_null(shapes_temp, self.pos)))
        code.put_gotref(shapes_temp, py_object_type)
        code.putln('%s = __pyx_array_new(%s, %s, PyBytes_AS_STRING(%s), (char *) "%s", (char *) %s); %s' % (self.result(), shapes_temp, itemsize, format_temp, self.mode, self.operand.result(), code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)

        def dispose(temp):
            code.put_decref_clear(temp, py_object_type)
            code.funcstate.release_temp(temp)
        dispose(shapes_temp)
        dispose(format_temp)

    @classmethod
    def from_carray(cls, src_node, env):
        """
        Given a C array type, return a CythonArrayNode
        """
        pos = src_node.pos
        base_type = src_node.type
        none_node = NoneNode(pos)
        axes = []
        while base_type.is_array:
            axes.append(SliceNode(pos, start=none_node, stop=none_node, step=none_node))
            base_type = base_type.base_type
        axes[-1].step = IntNode(pos, value='1', is_c_literal=True)
        memslicenode = Nodes.MemoryViewSliceTypeNode(pos, axes=axes, base_type_node=base_type)
        result = CythonArrayNode(pos, base_type_node=memslicenode, operand=src_node, array_dtype=base_type)
        result = result.analyse_types(env)
        return result