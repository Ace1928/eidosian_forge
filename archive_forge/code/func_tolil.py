import numpy
import cupy
from cupy import _core
from cupyx.scipy.sparse import _util
from cupyx.scipy.sparse import _sputils
def tolil(self, copy=False):
    """Convert this matrix to LInked List format."""
    return self.tocsr(copy=copy).tolil(copy=False)