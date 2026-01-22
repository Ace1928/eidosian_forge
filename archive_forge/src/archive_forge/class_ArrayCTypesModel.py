from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.ArrayCTypes)
class ArrayCTypesModel(StructModel):

    def __init__(self, dmm, fe_type):
        members = [('data', types.CPointer(fe_type.dtype)), ('meminfo', types.MemInfoPointer(fe_type.dtype))]
        super(ArrayCTypesModel, self).__init__(dmm, fe_type, members)