import os
import re
import types
from google.protobuf import message
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.protobuf import saved_metadata_pb2
from tensorflow.python.keras.protobuf import versions_pb2
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.saving.saved_model import utils
from tensorflow.python.keras.saving.saved_model.serialized_attributes import CommonEndpoints
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import load as tf_load
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.saved_model import revived_types
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import compat
from tensorflow.python.util import nest
def _unable_to_call_layer_due_to_serialization_issue(layer, *unused_args, **unused_kwargs):
    """Replaces the `layer.call` if the layer was not fully serialized.

  Keras Model/Layer serialization is relatively relaxed because SavedModels
  are not always loaded back as keras models. Thus, when there is an issue
  tracing a non-signature function, a warning is logged instead of raising an
  error. This results in a SavedModel where the model's call function is saved,
  but the internal layer call functions are not.

  When deserialized with `tf.keras.models.load_model`, the internal layers
  which do not have serialized call functions should raise an error when called.

  Args:
    layer: Layer without the serialized call function.

  Raises:
    ValueError
  """
    raise ValueError('Cannot call custom layer {} of type {}, because the call function was not serialized to the SavedModel.Please try one of the following methods to fix this issue:\n\n(1) Implement `get_config` and `from_config` in the layer/model class, and pass the object to the `custom_objects` argument when loading the model. For more details, see: https://www.tensorflow.org/guide/keras/save_and_serialize\n\n(2) Ensure that the subclassed model or layer overwrites `call` and not `__call__`. The input shape and dtype will be automatically recorded when the object is called, and used when saving. To manually specify the input shape/dtype, decorate the call function with `@tf.function(input_signature=...)`.'.format(layer.name, type(layer)))