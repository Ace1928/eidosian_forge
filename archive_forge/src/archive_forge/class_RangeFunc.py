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
class RangeFunc(BuiltinFunc):

    def __call__(self, *args, unroll=None):
        """Range with loop unrolling support.

        Args:
            start (int):
                Same as that of built-in :obj:`range`.
            stop (int):
                Same as that of built-in :obj:`range`.
            step (int):
                Same as that of built-in :obj:`range`.
            unroll (int or bool or None):

                - If `True`, add ``#pragma unroll`` directive before the
                  loop.
                - If `False`, add ``#pragma unroll(1)`` directive before
                  the loop to disable unrolling.
                - If an `int`, add ``#pragma unroll(n)`` directive before
                  the loop, where the integer ``n`` means the number of
                  iterations to unroll.
                - If `None` (default), leave the control of loop unrolling
                  to the compiler (no ``#pragma``).

        .. seealso:: `#pragma unroll`_

        .. _#pragma unroll:
            https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#pragma-unroll
        """
        super().__call__()

    def call(self, env, *args, unroll=None):
        if len(args) == 0:
            raise TypeError('range expected at least 1 argument, got 0')
        elif len(args) == 1:
            start, stop, step = (Constant(0), args[0], Constant(1))
        elif len(args) == 2:
            start, stop, step = (args[0], args[1], Constant(1))
        elif len(args) == 3:
            start, stop, step = args
        else:
            raise TypeError(f'range expected at most 3 argument, got {len(args)}')
        if unroll is not None:
            if not all((isinstance(x, Constant) for x in (start, stop, step, unroll))):
                raise TypeError('loop unrolling requires constant start, stop, step and unroll value')
            unroll = unroll.obj
            if not (isinstance(unroll, int) or isinstance(unroll, bool)):
                raise TypeError(f'unroll value expected to be of type int, got {type(unroll).__name__}')
            if unroll is False:
                unroll = 1
            if not (unroll is True or 0 < unroll < 1 << 31):
                warnings.warn('loop unrolling is ignored as the unroll value is non-positive or greater than INT_MAX')
        if isinstance(step, Constant):
            step_is_positive = step.obj >= 0
        elif step.ctype.dtype.kind == 'u':
            step_is_positive = True
        else:
            step_is_positive = None
        stop = Data.init(stop, env)
        start = Data.init(start, env)
        step = Data.init(step, env)
        if start.ctype.dtype.kind not in 'iu':
            raise TypeError('range supports only for integer type.')
        if stop.ctype.dtype.kind not in 'iu':
            raise TypeError('range supports only for integer type.')
        if step.ctype.dtype.kind not in 'iu':
            raise TypeError('range supports only for integer type.')
        if env.mode == 'numpy':
            ctype = _cuda_types.Scalar(int)
        elif env.mode == 'cuda':
            ctype = stop.ctype
        else:
            assert False
        return Range(start, stop, step, ctype, step_is_positive, unroll=unroll)