import tensorflow.compat.v2 as tf
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.dtensor import utils
from keras.src.engine.base_layer import Layer
from keras.src.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export
def _fused_can_be_used(self, ndims):
    """Returns false if fused implementation cannot be used.

        Check if the axis is contiguous and can be collapsed into the last axis.
        The self.axis is assumed to have no duplicates.
        """
    axis = sorted(self.axis)
    can_use_fused = False
    if axis[-1] == ndims - 1 and axis[-1] - axis[0] == len(axis) - 1:
        can_use_fused = True
    if self.epsilon < 1.001e-05 or self.dtype != 'float32':
        can_use_fused = False
    return can_use_fused