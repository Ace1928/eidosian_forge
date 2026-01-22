import numpy
import warnings
import cupy
from cupy.cuda import nccl
from cupyx.distributed import _store
from cupyx.distributed._comm import _Backend
from cupyx.scipy import sparse
def _check_contiguous(self, array):
    if not array.flags.c_contiguous and (not array.flags.f_contiguous):
        raise RuntimeError('NCCL requires arrays to be either c- or f-contiguous')