from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.tensorflow_stub import dtypes
def _norm_to_list_of_layers(maybe_layers):
    """Normalizes to a list of layers.

    Args:
      maybe_layers: A list of data[1] or a list of list of data.

    Returns:
      List of list of data.

    [1]: A Functional model has fields 'inbound_nodes' and 'output_layers' which can
    look like below:
    - ['in_layer_name', 0, 0]
    - [['in_layer_is_model', 1, 0], ['in_layer_is_model', 1, 1]]
    The data inside the list seems to describe [name, size, index].
    """
    return maybe_layers if isinstance(maybe_layers[0], (list,)) else [maybe_layers]