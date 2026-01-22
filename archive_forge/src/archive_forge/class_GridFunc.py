from typing import Any, Mapping
import warnings
import cupy
from cupy_backends.cuda.api import runtime
from cupy.cuda import device
from cupyx.jit import _cuda_types
from cupyx.jit import _cuda_typerules
from cupyx.jit._internal_types import BuiltinFunc
from cupyx.jit._internal_types import Data
from cupyx.jit._internal_types import Constant
from cupyx.jit._internal_types import Range
from cupyx.jit import _compile
from functools import reduce
class GridFunc(BuiltinFunc):

    def __init__(self, mode):
        if mode == 'grid':
            self._desc = 'Compute the thread index in the grid.'
            self._eq = 'jit.threadIdx.x + jit.blockIdx.x * jit.blockDim.x'
            self._link = 'numba.cuda.grid'
            self._code = 'threadIdx.{n} + blockIdx.{n} * blockDim.{n}'
        elif mode == 'gridsize':
            self._desc = 'Compute the grid size.'
            self._eq = 'jit.blockDim.x * jit.gridDim.x'
            self._link = 'numba.cuda.gridsize'
            self._code = 'blockDim.{n} * gridDim.{n}'
        else:
            raise ValueError('unsupported function')
        doc = f"        {self._desc}\n\n        Computation of the first integer is as follows::\n\n            {self._eq}\n\n        and for the other two integers the ``y`` and ``z`` attributes are used.\n\n        Args:\n            ndim (int): The dimension of the grid. Only 1, 2, or 3 is allowed.\n\n        Returns:\n            int or tuple:\n                If ``ndim`` is 1, an integer is returned, otherwise a tuple.\n\n        .. note::\n            This function follows the convention of Numba's\n            :func:`{self._link}`.\n        "
        self.__doc__ = doc

    def __call__(self, ndim):
        super().__call__()

    def call_const(self, env, ndim):
        if not isinstance(ndim, int):
            raise TypeError('ndim must be an integer')
        if ndim == 1:
            return Data(self._code.format(n='x'), _cuda_types.uint32)
        elif ndim == 2:
            dims = ('x', 'y')
        elif ndim == 3:
            dims = ('x', 'y', 'z')
        else:
            raise ValueError('Only ndim=1,2,3 are supported')
        elts_code = ', '.join((self._code.format(n=n) for n in dims))
        ctype = _cuda_types.Tuple([_cuda_types.uint32] * ndim)
        if ndim == 2:
            return Data(f'thrust::make_pair({elts_code})', ctype)
        else:
            return Data(f'thrust::make_tuple({elts_code})', ctype)