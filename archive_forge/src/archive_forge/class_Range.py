import functools
import itertools
from typing import Any, NoReturn, Optional, Union, TYPE_CHECKING
from cupyx.jit import _cuda_types
from cupyx.jit import _cuda_typerules
class Range(Expr):

    def __init__(self, start: Data, stop: Data, step: Data, ctype: _cuda_types.Scalar, step_is_positive: Optional[bool], *, unroll: Union[None, int, bool]=None) -> None:
        self.start = start
        self.stop = stop
        self.step = step
        self.ctype = ctype
        self.step_is_positive = step_is_positive
        self.unroll = unroll