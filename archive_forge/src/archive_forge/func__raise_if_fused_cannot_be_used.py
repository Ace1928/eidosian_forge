import warnings
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.dtensor import utils
from keras.src.engine.base_layer import Layer
from keras.src.engine.input_spec import InputSpec
from keras.src.utils import control_flow_util
from keras.src.utils import tf_utils
from tensorflow.python.ops.control_flow_ops import (
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import keras_export
def _raise_if_fused_cannot_be_used(self):
    """Raises a ValueError if fused implementation cannot be used.

        In addition to the checks done in this function, the input tensors rank
        must be 4 or 5. The input rank check can only be done once the input
        shape is known.
        """
    if self.renorm:
        raise ValueError('Passing both `fused=True` and `renorm=True` is not supported')
    axis = [self.axis] if isinstance(self.axis, int) else self.axis
    if len(axis) > 1 or axis[0] not in (-4, -3, -1, 1, 3, 4):
        raise ValueError('Passing `fused=True` is only supported when axis is 1 or 3 for input rank = 4 or 1 or 4 for input rank = 5. Got axis %s' % (axis,))
    if self.virtual_batch_size is not None:
        raise ValueError('Passing `fused=True` is not supported when `virtual_batch_size` is specified.')
    if self.adjustment is not None:
        raise ValueError('Passing `fused=True` is not supported when `adjustment` is specified.')
    if self._compute_dtype not in ('float16', 'bfloat16', 'float32', None):
        raise ValueError('Passing `fused=True` is only supported when the compute dtype is float16, bfloat16, or float32. Got dtype: %s' % (self._compute_dtype,))