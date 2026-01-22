import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
def _make_constant_bytes(context, builder, nbytes):
    bstr_ctor = cgutils.create_struct_proxy(bytes_type)
    bstr = bstr_ctor(context, builder)
    if isinstance(nbytes, int):
        nbytes = ir.Constant(bstr.nitems.type, nbytes)
    bstr.meminfo = context.nrt.meminfo_alloc(builder, nbytes)
    bstr.nitems = nbytes
    bstr.itemsize = ir.Constant(bstr.itemsize.type, 1)
    bstr.data = context.nrt.meminfo_data(builder, bstr.meminfo)
    bstr.parent = cgutils.get_null_value(bstr.parent.type)
    bstr.shape = cgutils.get_null_value(bstr.shape.type)
    bstr.strides = cgutils.get_null_value(bstr.strides.type)
    return bstr