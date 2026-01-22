import cupy
from cupyx.scipy._lib._util import _asarray_validated, float_factorial
def _set_dtype(self, dtype, union=False):
    if cupy.issubdtype(dtype, cupy.complexfloating) or cupy.issubdtype(self.dtype, cupy.complexfloating):
        self.dtype = cupy.complex_
    elif not union or self.dtype != cupy.complex_:
        self.dtype = cupy.float_