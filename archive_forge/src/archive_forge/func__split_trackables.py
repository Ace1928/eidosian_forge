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
def _split_trackables(trackable_data: List[_TrackableData]) -> Tuple[List[_TrackableData], List[_TrackableData], Dict[str, List[_TrackableData]]]:
    """Splits Trackables into 3 categories (tensor/pystate/registered)."""
    tensor_trackables = []
    pystate_trackables = []
    registered_trackables = collections.defaultdict(list)
    for td in trackable_data:
        saver_name = registration.get_registered_saver_name(td.object_to_save)
        if isinstance(td.object_to_save, python_state.PythonState):
            pystate_trackables.append(td)
        elif saver_name:
            registered_trackables[saver_name].append(td)
        else:
            tensor_trackables.append(td)
    return (tensor_trackables, pystate_trackables, registered_trackables)