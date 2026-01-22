import os
import warnings
from functools import partial
from math import ceil
from uuid import uuid4
import numpy as np
import pyarrow as pa
from multiprocess import get_context
from .. import config
class SharedMemoryContext:

    def __init__(self):
        self.created_shms = []
        self.opened_shms = []

    def get_shm(self, name, size, create):
        shm = SharedMemory(size=int(size), name=name, create=create)
        if create:
            self.created_shms.append(shm)
        else:
            self.opened_shms.append(shm)
        return shm

    def get_array(self, name, shape, dtype, create):
        shm = self.get_shm(name=name, size=np.prod(shape) * np.dtype(dtype).itemsize, create=create)
        return np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for shm in self.created_shms:
            shm.close()
            shm.unlink()
        for shm in self.opened_shms:
            shm.close()