import sys
from numba.cuda import cg
from .stubs import (threadIdx, blockIdx, blockDim, gridDim, laneid, warpsize,
from .intrinsics import (grid, gridsize, syncthreads, syncthreads_and,
from .cudadrv.error import CudaSupportError
from numba.cuda.cudadrv.driver import (BaseCUDAMemoryManager,
from numba.cuda.cudadrv.runtime import runtime
from .cudadrv import nvvm
from numba.cuda import initialize
from .errors import KernelRuntimeError
from .decorators import jit, declare_device
from .api import *
from .api import _auto_device
from .args import In, Out, InOut
from .intrinsic_wrapper import (all_sync, any_sync, eq_sync, ballot_sync,
from .kernels import reduction
Returns None if there was no error initializing the CUDA driver.
    If there was an error initializing the driver, a string describing the
    error is returned.
    