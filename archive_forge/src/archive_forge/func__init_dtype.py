import warnings
import cupy
from cupyx.scipy import sparse
from cupyx.scipy.sparse import _util
def _init_dtype(self):
    """Called from subclasses at the end of the `__init__` routine.
        """
    if self.dtype is None:
        v = cupy.zeros(self.shape[-1])
        self.dtype = self.matvec(v).dtype