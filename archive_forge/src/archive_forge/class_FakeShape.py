from contextlib import contextmanager
import numpy as np
from_record_like = None
class FakeShape(tuple):
    """
    The FakeShape class is used to provide a shape which does not allow negative
    indexing, similar to the shape in CUDA Python. (Numpy shape arrays allow
    negative indexing)
    """

    def __getitem__(self, k):
        if isinstance(k, int) and k < 0:
            raise IndexError('tuple index out of range')
        return super(FakeShape, self).__getitem__(k)