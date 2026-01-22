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
def _infer_inputs(self, layer_node_id, convert_to_shapes=False):
    """Infers input shape of layer from SavedModel functions."""
    call_fn_id = self._search_for_child_node(layer_node_id, ['call_and_return_all_conditional_losses'])
    if call_fn_id is None:
        return None
    concrete_functions = self._proto.nodes[call_fn_id].function.concrete_functions
    if not concrete_functions:
        return None
    call_fn_name = concrete_functions[0]
    call_fn_proto = self._proto.concrete_functions[call_fn_name]
    structured_input_signature = nested_structure_coder.decode_proto(call_fn_proto.canonicalized_input_signature)
    inputs = structured_input_signature[0][0]
    if convert_to_shapes:
        return nest.map_structure(lambda spec: spec.shape, inputs)
    else:
        return inputs