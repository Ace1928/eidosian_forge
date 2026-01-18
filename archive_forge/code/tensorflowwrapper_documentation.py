from typing import Any, Callable, Dict, Optional, Tuple, Type, TypeVar
import srsly
from ..compat import tensorflow as tf
from ..model import Model
from ..shims import TensorFlowShim, keras_model_fns, maybe_handshake_model
from ..types import ArgsKwargs, ArrayXd
from ..util import (
Return the output of the wrapped TensorFlow model for the given input,
    along with a callback to handle the backward pass.
    