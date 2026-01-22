from typing import Optional
import tensorflow as tf
from tensorflow import nest
from tensorflow.keras import layers
from autokeras.engine import block as block_module
from autokeras.utils import layer_utils
from autokeras.utils import utils
def _build_block(self, hp, output_node, reduction_type):
    if reduction_type == FLATTEN:
        output_node = Flatten().build(hp, output_node)
    elif reduction_type == GLOBAL_MAX:
        output_node = self.global_max(output_node)
    elif reduction_type == GLOBAL_AVG:
        output_node = self.global_avg(output_node)
    return output_node