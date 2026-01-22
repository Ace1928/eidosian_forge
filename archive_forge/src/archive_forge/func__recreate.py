import collections
import functools
import os
import sys
from absl import logging
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.function.capture import restore_captures
from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.checkpoint import restore
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.eager.polymorphic_function import saved_model_utils as function_saved_model_utils
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import fingerprinting
from tensorflow.python.saved_model import fingerprinting_utils
from tensorflow.python.saved_model import function_deserialization
from tensorflow.python.saved_model import load_options
from tensorflow.python.saved_model import load_v1_in_v2
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model import registration
from tensorflow.python.saved_model import revived_types
from tensorflow.python.saved_model import utils_impl as saved_model_utils
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base
from tensorflow.python.trackable import data_structures
from tensorflow.python.trackable import resource
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _recreate(self, proto, node_id, nodes):
    """Creates a Python object from a SavedObject protocol buffer.

    Args:
      proto: a SavedObject proto
      node_id: int, the index of this object in the SavedObjectGraph node list.
      nodes: dict mapping int node_ids -> created objects.

    Returns:
      The recreated object, and the set-attribute function for reconnecting
      the trackable children.
    """
    registered_class = registration.get_registered_class(proto.registered_name)
    if registered_class is None:
        registered_class = _BUILT_IN_REGISTRATIONS.get(proto.WhichOneof('kind'))
    dependencies = {}
    for key, dep_node_id in self._get_node_dependencies(proto).items():
        dependencies[key] = nodes[dep_node_id]
    if registered_class:
        obj = registered_class._deserialize_from_proto(proto=proto.serialized_user_proto, object_proto=proto, dependencies=dependencies, export_dir=self._export_dir, asset_file_def=self._asset_file_def, operation_attributes=self._operation_attributes)
        if isinstance(obj, base.Trackable):
            setter = type(obj)._add_trackable_child
        else:
            setter = setattr
        return (obj, setter)
    else:
        return self._recreate_default(proto, node_id, dependencies)