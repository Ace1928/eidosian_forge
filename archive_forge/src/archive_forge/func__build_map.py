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
def _build_map(outputs):
    """This method topologically sorts nodes in order from inputs to outputs.

  It uses a depth-first search to topologically sort nodes that appear in the
  _keras_history connectivity metadata of `outputs`.

  Args:
    outputs: the output tensors whose _keras_history metadata should be walked.
    This may be an arbitrary nested structure.

  Returns:
    A tuple like (ordered_nodes, layer_to_first_traversal_index)
    ordered_nodes: list of nodes appearing in the keras history, topologically
      sorted from original inputs to the `outputs`.
      (If outputs have different sets of ancestors, the inputs to one output
      may appear after a different output).
    layer_to_first_traversal_index:
      A dict mapping layer to the traversal index in the DFS where it is
      seen. Note: if a layer is shared by several nodes, the dict will only
      store the index corresponding to the *first* time the layer seen.
  """
    finished_nodes = set()
    nodes_in_progress = set()
    nodes_in_decreasing_depth = []
    layer_indices = {}
    for output in nest.flatten(outputs):
        _build_map_helper(output, finished_nodes, nodes_in_progress, nodes_in_decreasing_depth, layer_indices)
    return (nodes_in_decreasing_depth, layer_indices)