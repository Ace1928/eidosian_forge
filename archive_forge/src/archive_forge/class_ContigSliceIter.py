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
class ContigSliceIter(SliceIter):

    def start_loops(self):
        code = self.code
        code.begin_block()
        type_decl = self.slice_type.dtype.empty_declaration_code()
        total_size = ' * '.join(('%s.shape[%d]' % (self.slice_result, i) for i in range(self.ndim)))
        code.putln('Py_ssize_t __pyx_temp_extent = %s;' % total_size)
        code.putln('Py_ssize_t __pyx_temp_idx;')
        code.putln('%s *__pyx_temp_pointer = (%s *) %s.data;' % (type_decl, type_decl, self.slice_result))
        code.putln('for (__pyx_temp_idx = 0; __pyx_temp_idx < __pyx_temp_extent; __pyx_temp_idx++) {')
        return '__pyx_temp_pointer'

    def end_loops(self):
        self.code.putln('__pyx_temp_pointer += 1;')
        self.code.putln('}')
        self.code.end_block()