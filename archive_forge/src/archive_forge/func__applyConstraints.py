import warnings
import numpy
import cupy
import cupy.linalg as linalg
from cupyx.scipy.sparse import linalg as splinalg
def _applyConstraints(blockVectorV, YBY, blockVectorBY, blockVectorY):
    """Changes blockVectorV in place."""
    YBV = cupy.dot(blockVectorBY.T.conj(), blockVectorV)
    tmp = linalg.solve(YBY, YBV)
    blockVectorV -= cupy.dot(blockVectorY, tmp)