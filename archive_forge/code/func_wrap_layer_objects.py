import functools
import threading
import weakref
from tensorflow.python.eager import def_function
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.mixed_precision import autocast_variable
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import load as keras_load
from tensorflow.python.keras.saving.saved_model import serialized_attributes
from tensorflow.python.keras.saving.saved_model import utils
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
def wrap_layer_objects(layer, serialization_cache):
    """Returns extra trackable objects to attach to the serialized layer.

  Args:
    layer: Keras Layer object.
    serialization_cache: Dictionary shared between all objects during
      serialization.

  Returns:
    A dictionary containing all checkpointable objects from a
    SerializedAttributes object. See LayerAttributes and ModelAttributes for
    entire list of objects
  """
    all_losses = layer._callable_losses[:]
    for child_layer in utils.list_all_layers(layer):
        all_losses.extend(child_layer._callable_losses)
    keras_loss_cache = serialization_cache.setdefault('keras_losses', {})
    wrapped_loss_functions = []
    for loss_fn in all_losses:
        if loss_fn in keras_loss_cache:
            wrapped_loss_functions.append(keras_loss_cache[loss_fn])
        else:
            wrapped_loss = _wrap_unconditional_loss(loss_fn, len(keras_loss_cache))
            keras_loss_cache[loss_fn] = wrapped_loss
            wrapped_loss_functions.append(wrapped_loss)
    wrapped_layer_losses = [keras_loss_cache[fn] for fn in layer._callable_losses[:]]
    layer_metrics = data_structures.wrap_or_unwrap({m.name: m for m in layer._metrics})
    return dict(variables=data_structures.wrap_or_unwrap(layer.variables), trainable_variables=data_structures.wrap_or_unwrap(layer.trainable_variables), non_trainable_variables=data_structures.wrap_or_unwrap(layer.non_trainable_variables), layers=data_structures.wrap_or_unwrap(utils.list_all_layers(layer)), metrics=data_structures.wrap_or_unwrap(layer.metrics), regularization_losses=data_structures.wrap_or_unwrap(wrapped_loss_functions), layer_regularization_losses=data_structures.wrap_or_unwrap(wrapped_layer_losses), layer_metrics=layer_metrics)