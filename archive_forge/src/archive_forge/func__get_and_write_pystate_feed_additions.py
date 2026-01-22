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
def _get_and_write_pystate_feed_additions(pystate_trackables: List[_TrackableData], cache: Union[Dict[base.Trackable, Any], None], object_graph_proto=None) -> Tuple[Dict[base.Trackable, Any], Dict[base.Trackable, Any]]:
    """Gets feed additions needed for checkpointing Python State."""
    serialized_tensors = object_identity.ObjectIdentityDictionary()
    feed_additions = {}
    for td in pystate_trackables:
        trackable = td.object_to_save
        checkpoint_key = trackable_utils.checkpoint_key(td.object_name, python_state.PYTHON_STATE)
        if trackable in cache:
            save_string = cache[td.object_to_save][python_state.PYTHON_STATE]
        else:
            with ops.device('/cpu:0'):
                save_string = constant_op.constant('', dtype=dtypes.string)
                cache[trackable] = {python_state.PYTHON_STATE: save_string}
        with ops.init_scope():
            value = trackable.serialize()
        feed_additions[save_string] = value
        serialized_tensors[trackable] = {checkpoint_key: save_string}
        object_graph_proto.nodes[td.node_id].attributes.add(name=python_state.PYTHON_STATE, checkpoint_key=checkpoint_key, full_name=util.get_full_name(trackable))
    return (serialized_tensors, feed_additions)