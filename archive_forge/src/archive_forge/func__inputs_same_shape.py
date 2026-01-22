from typing import Optional
import tensorflow as tf
from tensorflow import nest
from tensorflow.keras import layers
from autokeras.engine import block as block_module
from autokeras.utils import layer_utils
from autokeras.utils import utils
def _inputs_same_shape(self, inputs):
    return all((input_node.shape.as_list() == inputs[0].shape.as_list() for input_node in inputs))