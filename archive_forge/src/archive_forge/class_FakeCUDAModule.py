from contextlib import contextmanager
import sys
import threading
import traceback
from numba.core import types
import numpy as np
from numba.np import numpy_support
from .vector_types import vector_types
class FakeCUDAModule(object):
    """
    An instance of this class will be injected into the __globals__ for an
    executing function in order to implement calls to cuda.*. This will fail to
    work correctly if the user code does::

        from numba import cuda as something_else

    In other words, the CUDA module must be called cuda.
    """

    def __init__(self, grid_dim, block_dim, dynshared_size):
        self.gridDim = Dim3(*grid_dim)
        self.blockDim = Dim3(*block_dim)
        self._cg = FakeCUDACg()
        self._local = FakeCUDALocal()
        self._shared = FakeCUDAShared(dynshared_size)
        self._const = FakeCUDAConst()
        self._atomic = FakeCUDAAtomic()
        self._fp16 = FakeCUDAFp16()
        for name, svty in vector_types.items():
            setattr(self, name, svty)
            for alias in svty.aliases:
                setattr(self, alias, svty)

    @property
    def cg(self):
        return self._cg

    @property
    def local(self):
        return self._local

    @property
    def shared(self):
        return self._shared

    @property
    def const(self):
        return self._const

    @property
    def atomic(self):
        return self._atomic

    @property
    def fp16(self):
        return self._fp16

    @property
    def threadIdx(self):
        return threading.current_thread().threadIdx

    @property
    def blockIdx(self):
        return threading.current_thread().blockIdx

    @property
    def warpsize(self):
        return 32

    @property
    def laneid(self):
        return threading.current_thread().thread_id % 32

    def syncthreads(self):
        threading.current_thread().syncthreads()

    def threadfence(self):
        pass

    def threadfence_block(self):
        pass

    def threadfence_system(self):
        pass

    def syncthreads_count(self, val):
        return threading.current_thread().syncthreads_count(val)

    def syncthreads_and(self, val):
        return threading.current_thread().syncthreads_and(val)

    def syncthreads_or(self, val):
        return threading.current_thread().syncthreads_or(val)

    def popc(self, val):
        return bin(val).count('1')

    def fma(self, a, b, c):
        return a * b + c

    def cbrt(self, a):
        return a ** (1 / 3)

    def brev(self, val):
        return int('{:032b}'.format(val)[::-1], 2)

    def clz(self, val):
        s = '{:032b}'.format(val)
        return len(s) - len(s.lstrip('0'))

    def ffs(self, val):
        s = '{:032b}'.format(val)
        r = (len(s) - len(s.rstrip('0')) + 1) % 33
        return r

    def selp(self, a, b, c):
        return b if a else c

    def grid(self, n):
        bdim = self.blockDim
        bid = self.blockIdx
        tid = self.threadIdx
        x = bid.x * bdim.x + tid.x
        if n == 1:
            return x
        y = bid.y * bdim.y + tid.y
        if n == 2:
            return (x, y)
        z = bid.z * bdim.z + tid.z
        if n == 3:
            return (x, y, z)
        raise RuntimeError('Global ID has 1-3 dimensions. %d requested' % n)

    def gridsize(self, n):
        bdim = self.blockDim
        gdim = self.gridDim
        x = bdim.x * gdim.x
        if n == 1:
            return x
        y = bdim.y * gdim.y
        if n == 2:
            return (x, y)
        z = bdim.z * gdim.z
        if n == 3:
            return (x, y, z)
        raise RuntimeError('Global grid has 1-3 dimensions. %d requested' % n)