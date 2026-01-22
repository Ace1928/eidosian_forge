from contextlib import contextmanager
import functools
import sys
import threading
import numpy as np
from .cudadrv.devicearray import FakeCUDAArray, FakeWithinKernelCUDAArray
from .kernelapi import Dim3, FakeCUDAModule, swapped_cuda_module
from ..errors import normalize_kernel_dimensions
from ..args import wrap_arg, ArgHint
class FakeCUDAKernel(object):
    """
    Wraps a @cuda.jit-ed function.
    """

    def __init__(self, fn, device, fastmath=False, extensions=[], debug=False):
        self.fn = fn
        self._device = device
        self._fastmath = fastmath
        self._debug = debug
        self.extensions = list(extensions)
        self.grid_dim = None
        self.block_dim = None
        self.stream = 0
        self.dynshared_size = 0
        functools.update_wrapper(self, fn)

    def __call__(self, *args):
        if self._device:
            with swapped_cuda_module(self.fn, _get_kernel_context()):
                return self.fn(*args)
        grid_dim, block_dim = normalize_kernel_dimensions(self.grid_dim, self.block_dim)
        fake_cuda_module = FakeCUDAModule(grid_dim, block_dim, self.dynshared_size)
        with _push_kernel_context(fake_cuda_module):
            retr = []

            def fake_arg(arg):
                _, arg = functools.reduce(lambda ty_val, extension: extension.prepare_args(*ty_val, stream=0, retr=retr), self.extensions, (None, arg))
                if isinstance(arg, np.ndarray) and arg.ndim > 0:
                    ret = wrap_arg(arg).to_device(retr)
                elif isinstance(arg, ArgHint):
                    ret = arg.to_device(retr)
                elif isinstance(arg, np.void):
                    ret = FakeCUDAArray(arg)
                else:
                    ret = arg
                if isinstance(ret, FakeCUDAArray):
                    return FakeWithinKernelCUDAArray(ret)
                return ret
            fake_args = [fake_arg(arg) for arg in args]
            with swapped_cuda_module(self.fn, fake_cuda_module):
                for grid_point in np.ndindex(*grid_dim):
                    bm = BlockManager(self.fn, grid_dim, block_dim, self._debug)
                    bm.run(grid_point, *fake_args)
            for wb in retr:
                wb()

    def __getitem__(self, configuration):
        self.grid_dim, self.block_dim = normalize_kernel_dimensions(*configuration[:2])
        if len(configuration) == 4:
            self.dynshared_size = configuration[3]
        return self

    def bind(self):
        pass

    def specialize(self, *args):
        return self

    def forall(self, ntasks, tpb=0, stream=0, sharedmem=0):
        if ntasks < 0:
            raise ValueError("Can't create ForAll with negative task count: %s" % ntasks)
        return self[ntasks, 1, stream, sharedmem]

    @property
    def overloads(self):
        return FakeOverloadDict()

    @property
    def py_func(self):
        return self.fn