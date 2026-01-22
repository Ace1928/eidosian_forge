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
def _insert_ancillary_layers(model, ancillary_layers, metrics_names, new_nodes):
    """Inserts ancillary layers into the model with the proper order."""
    metric_layers = [layer for layer in ancillary_layers if isinstance(layer, AddMetric)]
    metric_layers.sort(key=lambda layer: metrics_names.index(layer.metric_name))
    ancillary_layers = [layer for layer in ancillary_layers if not isinstance(layer, AddMetric)] + metric_layers
    model._insert_layers(ancillary_layers, relevant_nodes=list(new_nodes))