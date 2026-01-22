from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.NumpyNdIndexType)
class NdIndexModel(StructModel):

    def __init__(self, dmm, fe_type):
        ndim = fe_type.ndim
        members = [('shape', types.UniTuple(types.intp, ndim)), ('indices', types.EphemeralArray(types.intp, ndim)), ('exhausted', types.EphemeralPointer(types.boolean))]
        super(NdIndexModel, self).__init__(dmm, fe_type, members)