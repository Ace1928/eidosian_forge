import copy
import functools
import operator
import sys
import textwrap
import types as python_types
import warnings
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.ragged import ragged_getitem
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import get_canonical_name_for_symbol
from tensorflow.python.util.tf_export import get_symbol_from_name
class Masking(Layer):
    """Masks a sequence by using a mask value to skip timesteps.

  For each timestep in the input tensor (dimension #1 in the tensor),
  if all values in the input tensor at that timestep
  are equal to `mask_value`, then the timestep will be masked (skipped)
  in all downstream layers (as long as they support masking).

  If any downstream layer does not support masking yet receives such
  an input mask, an exception will be raised.

  Example:

  Consider a Numpy data array `x` of shape `(samples, timesteps, features)`,
  to be fed to an LSTM layer. You want to mask timestep #3 and #5 because you
  lack data for these timesteps. You can:

  - Set `x[:, 3, :] = 0.` and `x[:, 5, :] = 0.`
  - Insert a `Masking` layer with `mask_value=0.` before the LSTM layer:

  ```python
  samples, timesteps, features = 32, 10, 8
  inputs = np.random.random([samples, timesteps, features]).astype(np.float32)
  inputs[:, 3, :] = 0.
  inputs[:, 5, :] = 0.

  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Masking(mask_value=0.,
                                    input_shape=(timesteps, features)))
  model.add(tf.keras.layers.LSTM(32))

  output = model(inputs)
  # The time step 3 and 5 will be skipped from LSTM calculation.
  ```

  See [the masking and padding guide](
    https://www.tensorflow.org/guide/keras/masking_and_padding)
  for more details.
  """

    def __init__(self, mask_value=0.0, **kwargs):
        super(Masking, self).__init__(**kwargs)
        self.supports_masking = True
        self.mask_value = mask_value
        self._compute_output_and_mask_jointly = True

    def compute_mask(self, inputs, mask=None):
        return K.any(math_ops.not_equal(inputs, self.mask_value), axis=-1)

    def call(self, inputs):
        boolean_mask = K.any(math_ops.not_equal(inputs, self.mask_value), axis=-1, keepdims=True)
        outputs = inputs * math_ops.cast(boolean_mask, inputs.dtype)
        outputs._keras_mask = array_ops.squeeze(boolean_mask, axis=-1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'mask_value': self.mask_value}
        base_config = super(Masking, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))