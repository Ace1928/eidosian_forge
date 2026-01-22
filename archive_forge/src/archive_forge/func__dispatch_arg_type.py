import numpy
import warnings
import cupy
from cupy.cuda import nccl
from cupyx.distributed import _store
from cupyx.distributed._comm import _Backend
from cupyx.scipy import sparse
def _dispatch_arg_type(self, function, args):
    comm_class = _DenseNCCLCommunicator
    if isinstance(args[0], (list, tuple)) and sparse.issparse(args[0][0]) or sparse.issparse(args[0]):
        comm_class = _SparseNCCLCommunicator
    getattr(comm_class, function)(self, *args)