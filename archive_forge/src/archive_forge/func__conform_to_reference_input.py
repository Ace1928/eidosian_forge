import collections
import copy
import itertools
import warnings
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_layer as input_layer_module
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.engine import node as node_module
from tensorflow.python.keras.engine import training as training_lib
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.saving.saved_model import network_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls
def _conform_to_reference_input(self, tensor, ref_input):
    """Set shape and dtype based on `keras.Input`s."""
    if isinstance(tensor, tensor_lib.Tensor):
        t_shape = tensor.shape
        t_rank = t_shape.rank
        ref_shape = ref_input.shape
        ref_rank = ref_shape.rank
        keras_history = getattr(tensor, '_keras_history', None)
        if t_rank is not None and ref_rank is not None:
            if t_rank == ref_rank + 1 and t_shape[-1] == 1:
                tensor = array_ops.squeeze_v2(tensor, axis=-1)
            elif t_rank == ref_rank - 1 and ref_shape[-1] == 1:
                tensor = array_ops.expand_dims_v2(tensor, axis=-1)
        if keras_history is not None:
            tensor._keras_history = keras_history
        if not context.executing_eagerly():
            try:
                tensor.set_shape(tensor.shape.merge_with(ref_input.shape))
            except ValueError:
                logging.warning('Model was constructed with shape {} for input {}, but it was called on an input with incompatible shape {}.'.format(ref_input.shape, ref_input, tensor.shape))
        tensor = math_ops.cast(tensor, dtype=ref_input.dtype)
    elif tf_utils.is_extension_type(tensor):
        ref_input_dtype = getattr(ref_input, 'dtype', None)
        if ref_input_dtype is not None and ref_input_dtype != dtypes.variant:
            tensor = math_ops.cast(tensor, dtype=ref_input_dtype)
    return tensor