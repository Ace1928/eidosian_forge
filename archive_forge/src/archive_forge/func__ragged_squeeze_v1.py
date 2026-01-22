from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
@dispatch.dispatch_for_api(array_ops.squeeze)
def _ragged_squeeze_v1(input: ragged_tensor.Ragged, axis=None, name=None, squeeze_dims=None):
    axis = deprecation.deprecated_argument_lookup('axis', axis, 'squeeze_dims', squeeze_dims)
    return squeeze(input, axis, name)