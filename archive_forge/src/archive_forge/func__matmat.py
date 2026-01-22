import warnings
import cupy
from cupyx.scipy import sparse
from cupyx.scipy.sparse import _util
def _matmat(self, x):
    return x