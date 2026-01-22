from typing import Any, Callable, Dict, Optional, Tuple, Type, TypeVar
import srsly
from ..compat import tensorflow as tf
from ..model import Model
from ..shims import TensorFlowShim, keras_model_fns, maybe_handshake_model
from ..types import ArgsKwargs, ArrayXd
from ..util import (
def _convert_outputs(model, Ytf, is_train):
    Y = convert_recursive(is_tensorflow_array, tensorflow2xp, Ytf)

    def reverse_conversion(dY):
        return convert_recursive(is_xp_array, xp2tensorflow, dY)
    return (Y, reverse_conversion)