from __future__ import absolute_import
from .Errors import CompileError, error
from . import ExprNodes
from .ExprNodes import IntNode, NameNode, AttributeNode
from . import Options
from .Code import UtilityCode, TempitaUtilityCode
from .UtilityCode import CythonUtilityCode
from . import Buffer
from . import PyrexTypes
from . import ModuleNode
class StridedSliceIter(SliceIter):

    def start_loops(self):
        code = self.code
        code.begin_block()
        for i in range(self.ndim):
            t = (i, self.slice_result, i)
            code.putln('Py_ssize_t __pyx_temp_extent_%d = %s.shape[%d];' % t)
            code.putln('Py_ssize_t __pyx_temp_stride_%d = %s.strides[%d];' % t)
            code.putln('char *__pyx_temp_pointer_%d;' % i)
            code.putln('Py_ssize_t __pyx_temp_idx_%d;' % i)
        code.putln('__pyx_temp_pointer_0 = %s.data;' % self.slice_result)
        for i in range(self.ndim):
            if i > 0:
                code.putln('__pyx_temp_pointer_%d = __pyx_temp_pointer_%d;' % (i, i - 1))
            code.putln('for (__pyx_temp_idx_%d = 0; __pyx_temp_idx_%d < __pyx_temp_extent_%d; __pyx_temp_idx_%d++) {' % (i, i, i, i))
        return '__pyx_temp_pointer_%d' % (self.ndim - 1)

    def end_loops(self):
        code = self.code
        for i in range(self.ndim - 1, -1, -1):
            code.putln('__pyx_temp_pointer_%d += __pyx_temp_stride_%d;' % (i, i))
            code.putln('}')
        code.end_block()