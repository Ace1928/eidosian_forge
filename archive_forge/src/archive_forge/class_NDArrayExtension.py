import numpy as np
from ..core import Format
class NDArrayExtension(bsdf.Extension):
    """Copy of BSDF's NDArrayExtension but deal with lazy blobs."""
    name = 'ndarray'
    cls = np.ndarray

    def encode(self, s, v):
        return dict(shape=v.shape, dtype=str(v.dtype), data=v.tobytes())

    def decode(self, s, v):
        return v