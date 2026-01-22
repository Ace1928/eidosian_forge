from __future__ import absolute_import
from .PyrexTypes import CType, CTypedefType, CStructOrUnionType
import cython
@cython.cfunc
def _index_access(index_code, indices):
    indexing = ','.join([index_code(idx) for idx in indices])
    return ('[%s]' if len(indices) == 1 else '(%s)') % indexing