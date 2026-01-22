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
def _build_map_helper(tensor, finished_nodes, nodes_in_progress, nodes_in_decreasing_depth, layer_indices):
    """Recursive helper for `_build_map`."""
    layer, node_index, _ = tensor._keras_history
    node = layer._inbound_nodes[node_index]
    if node in finished_nodes:
        return
    if node in nodes_in_progress:
        raise ValueError('The tensor ' + str(tensor) + ' at layer "' + layer.name + '" is part of a cycle.')
    if layer not in layer_indices:
        layer_indices[layer] = len(layer_indices)
    nodes_in_progress.add(node)
    if not node.is_input:
        for tensor in node.keras_inputs:
            _build_map_helper(tensor, finished_nodes, nodes_in_progress, nodes_in_decreasing_depth, layer_indices)
    finished_nodes.add(node)
    nodes_in_progress.remove(node)
    nodes_in_decreasing_depth.append(node)