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
def _CompareForward(self, input_sizes, filter_sizes, output_sizes, stride, padding, dtype):
    x1 = np.random.rand(*input_sizes)
    x2 = np.random.rand(*filter_sizes)
    if isinstance(padding, list):
        padding = [(0, 0)] + padding + [(0, 0)]

    def _GetVal(use_gpu, dtype):
        with self.cached_session(use_gpu=use_gpu):
            t1 = constant_op.constant(x1, shape=input_sizes, dtype=dtype)
            t2 = constant_op.constant(x2, shape=filter_sizes, dtype=dtype)
            output = nn_ops.depthwise_conv2d_native(t1, t2, strides=[1, stride, stride, 1], padding=padding)
            ret = self.evaluate(output)
            self.assertShapeEqual(ret, output)
            return ret
    gpu_value = _GetVal(use_gpu=True, dtype=dtype)
    cpu_value = _GetVal(use_gpu=False, dtype=dtype)
    self.assertAllCloseAccordingToType(cpu_value, gpu_value, rtol=0.0001, atol=0.0001, bfloat16_rtol=0.1)