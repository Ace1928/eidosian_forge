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
def _search_for_child_node(self, parent_id, path_to_child):
    """Returns node id of child node.

    A helper method for traversing the object graph proto.

    As an example, say that the object graph proto in the SavedModel contains an
    object with the following child and grandchild attributes:

    `parent.child_a.child_b`

    This method can be used to retrieve the node id of `child_b` using the
    parent's node id by calling:

    `_search_for_child_node(parent_id, ['child_a', 'child_b'])`.

    Args:
      parent_id: node id of parent node
      path_to_child: list of children names.

    Returns:
      node_id of child, or None if child isn't found.
    """
    if not path_to_child:
        return parent_id
    for child in self._proto.nodes[parent_id].children:
        if child.local_name == path_to_child[0]:
            return self._search_for_child_node(child.node_id, path_to_child[1:])
    return None