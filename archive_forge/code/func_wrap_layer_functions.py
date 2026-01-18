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
def wrap_layer_functions(layer, serialization_cache):
    """Returns dict of wrapped layer call function and losses in tf.functions.

  Args:
    layer: Keras Layer object.
    serialization_cache: Dictionary shared between all objects during
      serialization.

  Returns:
    A dictionary containing all keras tf.functions to serialize. See
    LayerAttributes and ModelAttributes for the list of all attributes.
  """
    if isinstance(layer, keras_load.RevivedLayer) and (not isinstance(layer, sequential_lib.Sequential)):
        return {fn_name: getattr(layer.keras_api, fn_name, None) for fn_name in serialized_attributes.LayerAttributes.all_functions}
    original_fns = _replace_child_layer_functions(layer, serialization_cache)
    original_losses = _reset_layer_losses(layer)
    call_collection = LayerCallCollection(layer)
    call_fn_with_losses = call_collection.add_function(_wrap_call_and_conditional_losses(layer), '{}_layer_call_and_return_conditional_losses'.format(layer.name), match_layer_training_arg=True)
    call_fn = call_collection.add_function(_extract_outputs_from_fn(layer, call_fn_with_losses), '{}_layer_call_fn'.format(layer.name), match_layer_training_arg=False)
    fns = {'call_and_return_conditional_losses': call_fn_with_losses, '__call__': call_fn}
    if layer._activity_regularizer is not None:
        fns['activity_regularizer_fn'] = _wrap_activity_regularizer(layer)
        fns['call_and_return_all_conditional_losses'] = call_collection.add_function(_append_activity_regularizer_loss(layer, call_fn_with_losses, fns['activity_regularizer_fn']), '{}_layer_call_and_return_all_conditional_losses'.format(layer.name), match_layer_training_arg=False)
    else:
        fns['activity_regularizer_fn'] = None
        fns['call_and_return_all_conditional_losses'] = call_fn_with_losses
    with tracing_scope():
        call_collection.trace_with_input_signature()
        with base_layer_utils.call_context().enter(layer, inputs=None, build_graph=True, training=None, saving=True):
            for fn in fns.values():
                if fn is not None and fn.input_signature is not None:
                    if isinstance(fn, LayerCall):
                        fn = fn.wrapped_call
                    fn.get_concrete_function()
    _restore_child_layer_functions(original_fns)
    _restore_layer_losses(original_losses)
    return fns