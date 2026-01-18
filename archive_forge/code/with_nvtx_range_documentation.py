from typing import Any, Callable, Optional, Tuple, TypeVar
from ..model import Model
from ..util import use_nvtx_range
Wraps any layer and marks the forward and backprop phases as
    NVTX ranges for CUDA profiling.

    By default, the name of the layer is used as the name of the range,
    followed by the name of the pass.
    