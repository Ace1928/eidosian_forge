import collections
import warnings
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.layers.legacy_rnn import rnn_cell_wrapper_impl
from tensorflow.python.keras.legacy_tf_layers import base as base_layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['nn.rnn_cell.BasicRNNCell'])
class BasicRNNCell(LayerRNNCell):
    """The most basic RNN cell.

  Note that this cell is not optimized for performance. Please use
  `tf.contrib.cudnn_rnn.CudnnRNNTanh` for better performance on GPU.

  Args:
    num_units: int, The number of units in the RNN cell.
    activation: Nonlinearity to use.  Default: `tanh`. It could also be string
      that is within Keras activation function names.
    reuse: (optional) Python boolean describing whether to reuse variables in an
      existing scope.  If not `True`, and the existing scope already has the
      given variables, an error is raised.
    name: String, the name of the layer. Layers with the same name will share
      weights, but to avoid mistakes we require reuse=True in such cases.
    dtype: Default dtype of the layer (default of `None` means use the type of
      the first input). Required when `build` is called before `call`.
    **kwargs: Dict, keyword named properties for common layer attributes, like
      `trainable` etc when constructing the cell from configs of get_config().
  """

    def __init__(self, num_units, activation=None, reuse=None, name=None, dtype=None, **kwargs):
        warnings.warn('`tf.nn.rnn_cell.BasicRNNCell` is deprecated and will be removed in a future version. This class is equivalent as `tf.keras.layers.SimpleRNNCell`, and will be replaced by that in Tensorflow 2.0.')
        super(BasicRNNCell, self).__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)
        _check_supported_dtypes(self.dtype)
        if context.executing_eagerly() and tf_config.list_logical_devices('GPU'):
            logging.warning('%s: Note that this cell is not optimized for performance. Please use tf.contrib.cudnn_rnn.CudnnRNNTanh for better performance on GPU.', self)
        self.input_spec = input_spec.InputSpec(ndim=2)
        self._num_units = num_units
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
            raise ValueError('Expected inputs.shape[-1] to be known, saw shape: %s' % str(inputs_shape))
        _check_supported_dtypes(self.dtype)
        input_depth = inputs_shape[-1]
        self._kernel = self.add_variable(_WEIGHTS_VARIABLE_NAME, shape=[input_depth + self._num_units, self._num_units])
        self._bias = self.add_variable(_BIAS_VARIABLE_NAME, shape=[self._num_units], initializer=init_ops.zeros_initializer(dtype=self.dtype))
        self.built = True

    def call(self, inputs, state):
        """Most basic RNN: output = new_state = act(W * input + U * state + B)."""
        _check_rnn_cell_input_dtypes([inputs, state])
        gate_inputs = math_ops.matmul(array_ops.concat([inputs, state], 1), self._kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
        output = self._activation(gate_inputs)
        return (output, output)

    def get_config(self):
        config = {'num_units': self._num_units, 'activation': activations.serialize(self._activation), 'reuse': self._reuse}
        base_config = super(BasicRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))