import numpy as np
from numba.core import types, cgutils
from numba.core.datamodel import models
from numba.core.extending import (
from numba.core.imputils import impl_ret_borrowed
@unbox(SeriesType)
def unbox_series(typ, obj, c):
    """
    Convert a Series object to a native structure.
    """
    index = c.pyapi.object_getattr_string(obj, '_index')
    values = c.pyapi.object_getattr_string(obj, '_values')
    series = make_series(c.context, c.builder, typ)
    series.index = c.unbox(typ.index, index).value
    series.values = c.unbox(typ.values, values).value
    return NativeValue(series._getvalue())