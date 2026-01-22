from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
def _merge_function(self, inputs):
    base_layer_utils.no_ragged_support(inputs, self.name)
    if len(inputs) != 2:
        raise ValueError('A `Dot` layer should be called on exactly 2 inputs')
    x1 = inputs[0]
    x2 = inputs[1]
    if isinstance(self.axes, int):
        if self.axes < 0:
            axes = [self.axes % backend.ndim(x1), self.axes % backend.ndim(x2)]
        else:
            axes = [self.axes] * 2
    else:
        axes = []
        for i in range(len(self.axes)):
            if self.axes[i] < 0:
                axes.append(self.axes[i] % backend.ndim(inputs[i]))
            else:
                axes.append(self.axes[i])
    if self.normalize:
        x1 = nn.l2_normalize(x1, axis=axes[0])
        x2 = nn.l2_normalize(x2, axis=axes[1])
    output = backend.batch_dot(x1, x2, axes)
    return output