import functools
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers.pooling import AveragePooling1D
from tensorflow.python.keras.layers.pooling import AveragePooling2D
from tensorflow.python.keras.layers.pooling import AveragePooling3D
from tensorflow.python.keras.layers.pooling import MaxPooling1D
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.layers.pooling import MaxPooling3D
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
def _validate_init(self):
    if self.filters is not None and self.filters % self.groups != 0:
        raise ValueError('The number of filters must be evenly divisible by the number of groups. Received: groups={}, filters={}'.format(self.groups, self.filters))
    if not all(self.kernel_size):
        raise ValueError('The argument `kernel_size` cannot contain 0(s). Received: %s' % (self.kernel_size,))
    if not all(self.strides):
        raise ValueError('The argument `strides` cannot contains 0(s). Received: %s' % (self.strides,))
    if self.padding == 'causal' and (not isinstance(self, (Conv1D, SeparableConv1D))):
        raise ValueError('Causal padding is only supported for `Conv1D`and `SeparableConv1D`.')