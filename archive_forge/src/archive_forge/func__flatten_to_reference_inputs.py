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
def _flatten_to_reference_inputs(self, tensors):
    """Maps `tensors` to their respective `keras.Input`."""
    if self._enable_dict_to_input_mapping and isinstance(tensors, dict):
        ref_inputs = self._nested_inputs
        if not nest.is_nested(ref_inputs):
            ref_inputs = [self._nested_inputs]
        if isinstance(ref_inputs, dict):
            ref_input_names = sorted(ref_inputs.keys())
        else:
            ref_input_names = [inp._keras_history.layer.name for inp in ref_inputs]
        if len(tensors) > len(ref_input_names):
            warnings.warn('Input dict contained keys {} which did not match any model input. They will be ignored by the model.'.format([n for n in tensors.keys() if n not in ref_input_names]))
        try:
            return [tensors[n] for n in ref_input_names]
        except KeyError:
            return nest.flatten(tensors)
    return nest.flatten(tensors)