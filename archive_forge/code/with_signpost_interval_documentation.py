from typing import Any, Callable, Optional, Tuple, TypeVar
from ..compat import has_os_signpost, os_signpost
from ..model import Model
Wraps any layer and marks the init, forward and backprop phases using
    signpost intervals for macOS Instruments profiling

    By default, the name of the layer is used as the name of the range,
    followed by the name of the pass.
    