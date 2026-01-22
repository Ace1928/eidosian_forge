from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.NPDatetime)
@register_default(types.NPTimedelta)
class NPDatetimeModel(PrimitiveModel):

    def __init__(self, dmm, fe_type):
        be_type = ir.IntType(64)
        super(NPDatetimeModel, self).__init__(dmm, fe_type, be_type)