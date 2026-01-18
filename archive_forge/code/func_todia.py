import numpy
import cupy
from cupy import _core
from cupyx.scipy.sparse import _util
from cupyx.scipy.sparse import _sputils
def todia(self, copy=False):
    """Convert this matrix to sparse DIAgonal format."""
    return self.tocsr(copy=copy).todia(copy=False)