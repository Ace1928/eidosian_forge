import math
import operator
from functools import cached_property
from llvmlite import ir
from numba.core import types, typing, errors, cgutils
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.core.extending import overload_method, overload
from numba.misc import quicksort
from numba.cpython import slicing
from numba import literal_unroll
class ListPayloadAccessor(_ListPayloadMixin):
    """
    A helper object to access the list attributes given the pointer to the
    payload type.
    """

    def __init__(self, context, builder, list_type, payload_ptr):
        self._context = context
        self._builder = builder
        self._ty = list_type
        self._datamodel = context.data_model_manager[list_type.dtype]
        payload_type = types.ListPayload(list_type)
        ptrty = context.get_data_type(payload_type).as_pointer()
        payload_ptr = builder.bitcast(payload_ptr, ptrty)
        payload = context.make_data_helper(builder, payload_type, ref=payload_ptr)
        self._payload = payload