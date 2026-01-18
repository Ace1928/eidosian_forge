from collections import namedtuple
from textwrap import indent
from numba.types import float32, float64, int16, int32, int64, void, Tuple
from numba.core.typing.templates import signature

    Given the return type and arguments for a libdevice function, return the
    signature of the stub function used to call it from CUDA Python.
    