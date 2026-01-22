from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
def _validate_call_args(self, inputs, mask):
    """Validates arguments of the call method."""
    class_name = self.__class__.__name__
    if not isinstance(inputs, list):
        raise ValueError('{} layer must be called on a list of inputs, namely [query, value] or [query, value, key].'.format(class_name))
    if len(inputs) < 2 or len(inputs) > 3:
        raise ValueError('{} layer accepts inputs list of length 2 or 3, namely [query, value] or [query, value, key]. Given length: {}'.format(class_name, len(inputs)))
    if mask:
        if not isinstance(mask, list):
            raise ValueError('{} layer mask must be a list, namely [query_mask, value_mask].'.format(class_name))
        if len(mask) < 2 or len(mask) > len(inputs):
            raise ValueError('{} layer mask must be a list of length 2, namely [query_mask, value_mask]. Given length: {}'.format(class_name, len(mask)))