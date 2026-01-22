from typing import Optional
import tensorflow as tf
from tensorflow import nest
from tensorflow.keras import layers
from autokeras.engine import block as block_module
from autokeras.utils import layer_utils
from autokeras.utils import utils
class TemporalReduction(Reduction):
    """Reduce the dimension of a temporal tensor, e.g. output of RNN, to a vector.

    # Arguments
        reduction_type: String. 'flatten', 'global_max' or 'global_avg'. If left
            unspecified, it will be tuned automatically.
    """

    def __init__(self, reduction_type: Optional[str]=None, **kwargs):
        super().__init__(reduction_type, **kwargs)

    def global_max(self, input_node):
        return tf.math.reduce_max(input_node, axis=-2)

    def global_avg(self, input_node):
        return tf.math.reduce_mean(input_node, axis=-2)