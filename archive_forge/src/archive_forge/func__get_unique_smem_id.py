from functools import reduce
import operator
import math
from llvmlite import ir
import llvmlite.binding as ll
from numba.core.imputils import Registry, lower_cast
from numba.core.typing.npydecl import parse_dtype
from numba.core.datamodel import models
from numba.core import types, cgutils
from numba.np import ufunc_db
from numba.np.npyimpl import register_ufuncs
from .cudadrv import nvvm
from numba import cuda
from numba.cuda import nvvmutils, stubs, errors
from numba.cuda.types import dim3, CUDADispatcher
def _get_unique_smem_id(name):
    """Due to bug with NVVM invalid internalizing of shared memory in the
    PTX output.  We can't mark shared memory to be internal. We have to
    ensure unique name is generated for shared memory symbol.
    """
    global _unique_smem_id
    _unique_smem_id += 1
    return '{0}_{1}'.format(name, _unique_smem_id)