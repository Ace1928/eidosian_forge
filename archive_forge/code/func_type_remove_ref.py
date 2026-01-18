from __future__ import absolute_import
from .PyrexTypes import CType, CTypedefType, CStructOrUnionType
import cython
@cython.cfunc
def type_remove_ref(ty):
    return 'typename std::remove_reference<%s>::type' % ty