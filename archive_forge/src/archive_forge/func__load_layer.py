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
def _load_layer(self, node_id, identifier, metadata):
    """Load a single layer from a SavedUserObject proto."""
    metadata = json_utils.decode(metadata)
    if node_id in self.loaded_nodes:
        node, setter = self.loaded_nodes[node_id]
        _maybe_add_serialized_attributes(node, metadata)
        config = metadata.get('config')
        if _is_graph_network(node) and generic_utils.validate_config(config):
            child_nodes = self._get_child_layer_node_ids(node_id)
            self.model_layer_dependencies[node_id] = (node, child_nodes)
            if not child_nodes:
                self._models_to_reconstruct.append(node_id)
        return (node, setter)
    obj, setter = self._revive_from_config(identifier, metadata, node_id)
    if obj is None:
        obj, setter = revive_custom_object(identifier, metadata)
    _maybe_add_serialized_attributes(obj, metadata)
    return (obj, setter)