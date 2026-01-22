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
class LaneID(BuiltinFunc):

    def __call__(self):
        """Returns the lane ID of the calling thread, ranging in
        ``[0, jit.warpsize)``.

        .. note::
            Unlike :obj:`numba.cuda.laneid`, this is a callable function
            instead of a property.
        """
        super().__call__()

    def _get_preamble(self):
        preamble = '__device__ __forceinline__ unsigned int LaneId() {'
        if not runtime.is_hip:
            preamble += '\n                unsigned int ret;\n                asm ("mov.u32 %0, %%laneid;" : "=r"(ret) );\n                return ret; }\n            '
        else:
            preamble += '\n                return __lane_id(); }\n            '
        return preamble

    def call_const(self, env):
        env.generated.add_code(self._get_preamble())
        return Data('LaneId()', _cuda_types.uint32)