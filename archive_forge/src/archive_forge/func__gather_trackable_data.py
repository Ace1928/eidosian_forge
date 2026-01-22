import collections
from typing import Any, Callable, List, Optional, Tuple, Mapping, Union, Dict
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.checkpoint import graph_view as graph_view_lib
from tensorflow.python.checkpoint import save_util_v1
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.checkpoint import util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import base
from tensorflow.python.trackable import python_state
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import saveable_object as saveable_object_lib
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.types import core
from tensorflow.python.util import object_identity
def _gather_trackable_data(graph_view: graph_view_lib.ObjectGraphView, object_map: Mapping[base.Trackable, base.Trackable]) -> Tuple[List[_TrackableData], Dict[base.Trackable, int]]:
    """Returns a list of generated TrackableData based on the ObjectGraphView."""
    trackable_objects, node_paths = graph_view.breadth_first_traversal()
    object_names = object_identity.ObjectIdentityDictionary()
    for obj, path in node_paths.items():
        object_names[obj] = trackable_utils.object_path_to_string(path)
    node_ids = object_identity.ObjectIdentityDictionary()
    for node_id, node in enumerate(trackable_objects):
        node_ids[node] = node_id
    slot_variables = util.serialize_slot_variables(trackable_objects=trackable_objects, node_ids=node_ids, object_names=object_names)
    trackable_data = []
    for trackable in trackable_objects:
        children_proto = []
        for child in graph_view.list_children(trackable):
            children_proto.append(trackable_object_graph_pb2.TrackableObjectGraph.TrackableObject.ObjectReference(node_id=node_ids[child.ref], local_name=child.name))
        trackable_data.append(_TrackableData(trackable, node_id=node_ids[trackable], object_name=object_names[trackable], children_proto=children_proto, slot_variable_proto=slot_variables.get(trackable, []), object_to_save=util.get_mapped_trackable(trackable, object_map)))
    return (trackable_data, node_ids)