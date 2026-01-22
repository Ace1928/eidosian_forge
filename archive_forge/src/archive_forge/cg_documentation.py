from cupy.cuda import runtime as _runtime
from cupyx.jit import _compile
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import BuiltinFunc as _BuiltinFunc
from cupyx.jit._internal_types import Constant as _Constant
from cupyx.jit._internal_types import Data as _Data
from cupyx.jit._internal_types import wraps_class_method as _wraps_class_method
Calls ``cg::wait_prior<N>()``.

        Args:
            group: a valid cooperative group
            step (int): wait for the first ``N`` steps to finish

        .. seealso: `cg::wait_prior`_

        .. _cg::wait_prior:
            https://docs.nvidia.com/cuda/archive/11.6.0/cuda-c-programming-guide/index.html#collectives-cg-wait
        