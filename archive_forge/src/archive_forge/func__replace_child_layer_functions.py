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
def _replace_child_layer_functions(layer, serialization_cache):
    """Replaces functions in the children layers with wrapped tf.functions.

  This step allows functions from parent layers to reference the wrapped
  functions from their children layers instead of retracing the ops.

  This function also resets all losses stored in the layer. These are stored in
  the returned dictionary. Use `_restore_child_layer_functions` to restore
  the original attributes.

  Args:
    layer: Keras Layer object.
    serialization_cache: Dictionary shared between all objects during
      serialization.

  Returns:
    Dictionary mapping layer objects -> original functions and losses:
      { Child layer 1: {
          'losses': Original losses,
          'call': Original call function
          '_activity_regularizer': Original activity regularizer},
        Child layer 2: ...
      }
  """
    original_fns = {}

    def replace_layer_functions(child_layer, serialized_fns):
        """Replaces layer call and activity regularizer with wrapped functions."""
        original_fns[child_layer] = {'call': child_layer.call, '_activity_regularizer': child_layer._activity_regularizer}
        with utils.no_automatic_dependency_tracking_scope(child_layer):
            try:
                child_layer._activity_regularizer = serialized_fns.get('activity_regularizer_fn')
            except AttributeError:
                pass
            child_layer.call = utils.use_wrapped_call(child_layer, serialized_fns['call_and_return_conditional_losses'], default_training_value=False)

    def replace_metric_functions(child_layer, serialized_fns):
        """Replaces metric functions with wrapped functions."""
        original_fns[child_layer] = {'__call__': child_layer.__call__, 'result': child_layer.result, 'update_state': child_layer.update_state}
        with utils.no_automatic_dependency_tracking_scope(child_layer):
            child_layer.__call__ = serialized_fns['__call__']
            child_layer.result = serialized_fns['result']
            child_layer.update_state = serialized_fns['update_state']
    for child_layer in utils.list_all_layers(layer):
        if isinstance(child_layer, input_layer.InputLayer):
            continue
        if child_layer not in serialization_cache[constants.KERAS_CACHE_KEY]:
            serialized_functions = child_layer._trackable_saved_model_saver._get_serialized_attributes(serialization_cache).functions
        else:
            serialized_functions = serialization_cache[constants.KERAS_CACHE_KEY][child_layer].functions
        if not serialized_functions:
            continue
        if isinstance(child_layer, metrics.Metric):
            replace_metric_functions(child_layer, serialized_functions)
        else:
            replace_layer_functions(child_layer, serialized_functions)
    return original_fns