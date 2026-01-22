from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
class StructRefModel(StructModel):
    """Model for a mutable struct.
    A reference to the payload
    """

    def __init__(self, dmm, fe_typ):
        dtype = fe_typ.get_data_type()
        members = [('meminfo', types.MemInfoPointer(dtype))]
        super().__init__(dmm, fe_typ, members)