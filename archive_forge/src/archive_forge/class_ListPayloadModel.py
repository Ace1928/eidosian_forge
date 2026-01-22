from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.ListPayload)
class ListPayloadModel(StructModel):

    def __init__(self, dmm, fe_type):
        members = [('size', types.intp), ('allocated', types.intp), ('dirty', types.boolean), ('data', fe_type.container.dtype)]
        super(ListPayloadModel, self).__init__(dmm, fe_type, members)