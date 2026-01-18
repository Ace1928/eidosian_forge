import numpy
import cupy
from cupy import _core
from cupyx.scipy.sparse import _util
from cupyx.scipy.sparse import _sputils
def todense(self, order=None, out=None):
    """Return a dense matrix representation of this matrix."""
    return self.toarray(order=order, out=out)