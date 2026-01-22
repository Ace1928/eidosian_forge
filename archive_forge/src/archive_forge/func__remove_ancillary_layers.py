from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras.engine import functional
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.engine import training_v1
from tensorflow.python.keras.engine.base_layer import AddMetric
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.saving import model_config
from tensorflow.python.keras.saving import save
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
def _remove_ancillary_layers(model, layer_map, layers):
    """Removes and returns any ancillary layers from `layers` based on `model`.

  Ancillary layers are part of the model topology but not used to compute the
  model outputs, e.g., layers from `add_loss` and `add_metric`.

  Args:
    model: A Keras Model.
    layer_map: A map to from layers in the `model` to those in `layers`.
    layers: A list of all layers.

  Returns:
    Two lists of layers: (1) `layers` with the ancillary layers removed, and (2)
    the ancillary layers.
  """
    ancillary_layers = []
    if not model._is_graph_network:
        return (layers, ancillary_layers)
    depths = [depth for depth in model._nodes_by_depth.keys() if depth < 0]
    depths.sort(reverse=True)
    for depth in depths:
        for node in model._nodes_by_depth[depth]:
            ancillary_layers.append(layer_map[node.outbound_layer])
    return ([l for l in layers if l not in ancillary_layers], ancillary_layers)