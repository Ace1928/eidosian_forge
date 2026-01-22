import cupy
from cupy_backends.cuda.api import runtime
from cupy import _util
from cupyx.scipy.ndimage import _filters_core
def _get_type_info(param, dtype, types):
    if param.dtype is not None:
        return param.ctype
    ctype = cupy._core._scalar.get_typename(dtype)
    types.setdefault(param.ctype, ctype)
    return ctype