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
def _map_graph_network(inputs, outputs):
    """Validates a network's topology and gather its layers and nodes.

  Args:
    inputs: List of input tensors.
    outputs: List of outputs tensors.

  Returns:
    A tuple `(nodes, nodes_by_depth, layers, layers_by_depth)`.
    - nodes: list of Node instances.
    - nodes_by_depth: dict mapping ints (depth) to lists of node instances.
    - layers: list of Layer instances.
    - layers_by_depth: dict mapping ints (depth) to lists of layer instances.

  Raises:
    ValueError: In case the network is not valid (e.g. disconnected graph).
  """
    nodes_in_decreasing_depth, layer_indices = _build_map(outputs)
    network_nodes = {_make_node_key(node.layer.name, node.layer._inbound_nodes.index(node)) for node in nodes_in_decreasing_depth}
    nodes_depths = {}
    layers_depths = {}
    for node in reversed(nodes_in_decreasing_depth):
        depth = nodes_depths.setdefault(node, 0)
        previous_depth = layers_depths.get(node.layer, 0)
        depth = max(depth, previous_depth)
        layers_depths[node.layer] = depth
        nodes_depths[node] = depth
        for node_dep in node.parent_nodes:
            previous_depth = nodes_depths.get(node_dep, 0)
            nodes_depths[node_dep] = max(depth + 1, previous_depth)
    for input_t in inputs:
        input_layer = input_t._keras_history[0]
        if input_layer not in layers_depths:
            layers_depths[input_layer] = 0
            layer_indices[input_layer] = -1
            nodes_depths[input_layer._inbound_nodes[0]] = 0
            network_nodes.add(_make_node_key(input_layer.name, 0))
    nodes_by_depth = collections.defaultdict(list)
    for node, depth in nodes_depths.items():
        nodes_by_depth[depth].append(node)
    layers_by_depth = collections.defaultdict(list)
    for layer, depth in layers_depths.items():
        layers_by_depth[depth].append(layer)
    depth_keys = list(layers_by_depth.keys())
    depth_keys.sort(reverse=True)
    layers = []
    for depth in depth_keys:
        layers_for_depth = layers_by_depth[depth]
        layers_for_depth.sort(key=lambda x: layer_indices[x])
        layers.extend(layers_for_depth)
    depth_keys = list(nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    computable_tensors = set()
    for x in inputs:
        computable_tensors.add(id(x))
    layers_with_complete_input = []
    for depth in depth_keys:
        for node in nodes_by_depth[depth]:
            layer = node.layer
            if layer and (not node.is_input):
                for x in nest.flatten(node.keras_inputs):
                    if id(x) not in computable_tensors:
                        raise ValueError('Graph disconnected: cannot obtain value for tensor ' + str(x) + ' at layer "' + layer.name + '". The following previous layers were accessed without issue: ' + str(layers_with_complete_input))
                for x in nest.flatten(node.outputs):
                    computable_tensors.add(id(x))
                layers_with_complete_input.append(layer.name)
    all_names = [layer.name for layer in layers]
    for name in all_names:
        if all_names.count(name) != 1:
            raise ValueError('The name "' + name + '" is used ' + str(all_names.count(name)) + ' times in the model. All layer names should be unique.')
    return (network_nodes, nodes_by_depth, layers, layers_by_depth)