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
def _add_children_recreated_from_config(self, obj, proto, node_id):
    """Recursively records objects recreated from config."""
    if node_id in self._traversed_nodes_from_config:
        return
    parent_path = self._node_paths[node_id]
    self._traversed_nodes_from_config.add(node_id)
    obj._maybe_initialize_trackable()
    if isinstance(obj, base_layer.Layer) and (not obj.built):
        metadata = json_utils.decode(self._metadata[node_id].metadata)
        self._try_build_layer(obj, node_id, metadata.get('build_input_shape'))
    children = []
    for reference in proto.children:
        obj_child = obj._lookup_dependency(reference.local_name)
        children.append((obj_child, reference.node_id, reference.local_name))
    metric_list_node_id = self._search_for_child_node(node_id, [constants.KERAS_ATTR, 'layer_metrics'])
    if metric_list_node_id is not None and hasattr(obj, '_metrics'):
        obj_metrics = {m.name: m for m in obj._metrics}
        for reference in self._proto.nodes[metric_list_node_id].children:
            metric = obj_metrics.get(reference.local_name)
            if metric is not None:
                metric_path = '{}.layer_metrics.{}'.format(constants.KERAS_ATTR, reference.local_name)
                children.append((metric, reference.node_id, metric_path))
    for obj_child, child_id, child_name in children:
        child_proto = self._proto.nodes[child_id]
        if not isinstance(obj_child, trackable.Trackable):
            continue
        if child_proto.user_object.identifier in revived_types.registered_identifiers():
            setter = revived_types.get_setter(child_proto.user_object)
        elif obj_child._object_identifier in constants.KERAS_OBJECT_IDENTIFIERS:
            setter = _revive_setter
        else:
            setter = setattr
        if child_id in self.loaded_nodes:
            if self.loaded_nodes[child_id][0] is not obj_child:
                logging.warning('Looks like there is an object (perhaps variable or layer) that is shared between different layers/models. This may cause issues when restoring the variable values. Object: {}'.format(obj_child))
            continue
        if child_proto.WhichOneof('kind') == 'variable' and child_proto.variable.name:
            obj_child._handle_name = child_proto.variable.name + ':0'
        if isinstance(obj_child, data_structures.TrackableDataStructure):
            setter = lambda *args: None
        child_path = '{}.{}'.format(parent_path, child_name)
        self._node_paths[child_id] = child_path
        self._add_children_recreated_from_config(obj_child, child_proto, child_id)
        self.loaded_nodes[child_id] = (obj_child, setter)