from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.Integer)
@register_default(types.IntegerLiteral)
class IntegerModel(PrimitiveModel):

    def __init__(self, dmm, fe_type):
        be_type = ir.IntType(fe_type.bitwidth)
        super(IntegerModel, self).__init__(dmm, fe_type, be_type)