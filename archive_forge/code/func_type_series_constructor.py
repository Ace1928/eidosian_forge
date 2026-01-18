import numpy as np
from numba.core import types, cgutils
from numba.core.datamodel import models
from numba.core.extending import (
from numba.core.imputils import impl_ret_borrowed
@type_callable(Series)
def type_series_constructor(context):

    def typer(data, index):
        if isinstance(index, IndexType) and isinstance(data, types.Array):
            assert data.layout == 'C'
            assert data.ndim == 1
            return SeriesType(data.dtype, index)
    return typer