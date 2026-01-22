import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
def _VerifyHandValues(self, tensor_in_sizes, filter_in_sizes, stride, padding, expected, use_gpu):
    """Verifies the output values of the depthwise convolution function.

    Args:
      tensor_in_sizes: Input tensor dimensions in [batch, input_rows,
        input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in [filter_rows, filter_cols,
        input_depth, depth_multiplier].
      stride: Stride.
      padding: Padding type.
      expected: An array containing the expected operation outputs.
      use_gpu: Whether to use GPU.
    """
    total_size_1 = 1
    total_size_2 = 1
    for s in tensor_in_sizes:
        total_size_1 *= s
    for s in filter_in_sizes:
        total_size_2 *= s
    x1 = [f * 1.0 for f in range(1, total_size_1 + 1)]
    x2 = [f * 1.0 for f in range(1, total_size_2 + 1)]
    with self.cached_session(use_gpu=use_gpu) as sess:
        t1 = constant_op.constant(x1, shape=tensor_in_sizes)
        t1.set_shape(tensor_in_sizes)
        t2 = constant_op.constant(x2, shape=filter_in_sizes)
        conv = nn_ops.depthwise_conv2d_native(t1, t2, strides=[1, stride, stride, 1], padding=padding)
        value = self.evaluate(conv)
    tf_logging.info('value = %r', value)
    self.assertArrayNear(expected, np.ravel(value), 1e-05)
    self.assertShapeEqual(value, conv)