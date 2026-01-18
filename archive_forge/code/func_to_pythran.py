from __future__ import absolute_import
from .PyrexTypes import CType, CTypedefType, CStructOrUnionType
import cython
@cython.ccall
def to_pythran(op, ptype=None):
    op_type = op.type
    if op_type.is_int:
        return op_type.cast_code(op.result())
    if is_type(op_type, ['is_pythran_expr', 'is_numeric', 'is_float', 'is_complex']):
        return op.result()
    if op.is_none:
        return 'pythonic::%s::None' % pythran_builtins
    if ptype is None:
        ptype = pythran_type(op_type)
    assert op.type.is_pyobject
    return 'from_python<%s>(%s)' % (ptype, op.py_result())