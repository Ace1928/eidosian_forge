import warnings
import cupy
from cupyx.scipy import sparse
from cupyx.scipy.sparse import _util
def _power(self, fun, x):
    res = cupy.array(x, copy=True)
    for i in range(self.args[1]):
        res = fun(res)
    return res