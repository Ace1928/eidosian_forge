from typing import Any, Callable, Optional, Tuple, Type
from ..config import registry
from ..model import Model
from ..shims import MXNetShim
from ..types import ArgsKwargs
from ..util import convert_recursive, is_mxnet_array, is_xp_array, mxnet2xp, xp2mxnet
Return the output of the wrapped MXNet model for the given input,
    along with a callback to handle the backward pass.
    