from typing import Any, Callable, Dict, Optional, Tuple, Type, TypeVar
import srsly
from ..compat import tensorflow as tf
from ..model import Model
from ..shims import TensorFlowShim, keras_model_fns, maybe_handshake_model
from ..types import ArgsKwargs, ArrayXd
from ..util import (
def TensorFlowWrapper(tensorflow_model: Any, convert_inputs: Optional[Callable]=None, convert_outputs: Optional[Callable]=None, optimizer: Optional[Any]=None, model_class: Type[Model]=Model, model_name: str='tensorflow') -> Model[InT, OutT]:
    """Wrap a TensorFlow model, so that it has the same API as Thinc models.
    To optimize the model, you'll need to create a TensorFlow optimizer and call
    optimizer.apply_gradients after each batch.
    """
    assert_tensorflow_installed()
    if not isinstance(tensorflow_model, tf.keras.models.Model):
        err = f'Expected tf.keras.models.Model, got: {type(tensorflow_model)}'
        raise ValueError(err)
    tensorflow_model = maybe_handshake_model(tensorflow_model)
    if convert_inputs is None:
        convert_inputs = _convert_inputs
    if convert_outputs is None:
        convert_outputs = _convert_outputs
    return model_class(model_name, forward, shims=[TensorFlowShim(tensorflow_model, optimizer=optimizer)], attrs={'convert_inputs': convert_inputs, 'convert_outputs': convert_outputs})