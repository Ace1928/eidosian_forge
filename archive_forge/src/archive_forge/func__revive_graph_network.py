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
def _revive_graph_network(self, identifier, metadata, node_id):
    """Revives a graph network from config."""
    config = metadata.get('config')
    if not generic_utils.validate_config(config):
        return None
    class_name = compat.as_str(metadata['class_name'])
    if generic_utils.get_registered_object(class_name) is not None:
        return None
    model_is_functional_or_sequential = metadata.get('is_graph_network', False) or class_name == 'Sequential' or class_name == 'Functional'
    if not model_is_functional_or_sequential:
        return None
    if class_name == 'Sequential':
        model = models_lib.Sequential(name=config['name'])
    elif identifier == constants.SEQUENTIAL_IDENTIFIER:
        model = models_lib.Sequential(name=class_name)
    else:
        model = models_lib.Functional(inputs=[], outputs=[], name=config['name'])
    layers = self._get_child_layer_node_ids(node_id)
    self.model_layer_dependencies[node_id] = (model, layers)
    if not layers:
        self._models_to_reconstruct.append(node_id)
    return model