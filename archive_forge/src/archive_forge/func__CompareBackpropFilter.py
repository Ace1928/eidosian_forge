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
def _CompareBackpropFilter(self, input_sizes, filter_sizes, output_sizes, stride, padding, dtype):
    x0 = np.random.rand(*input_sizes)
    x2 = np.random.rand(*output_sizes)
    padding_nhwc = padding
    padding_nchw = padding
    if isinstance(padding, list):
        padding_nhwc = [(0, 0)] + padding + [(0, 0)]
        padding_nchw = [(0, 0)] + [(0, 0)] + padding

    def _GetVal(use_gpu, dtype, data_format='NHWC'):
        with self.cached_session(use_gpu=use_gpu):
            t0 = constant_op.constant(x0, shape=input_sizes, dtype=dtype)
            t1 = constant_op.constant(filter_sizes, shape=[len(filter_sizes)])
            t2 = constant_op.constant(x2, shape=output_sizes, dtype=dtype)
            strides = [1, stride, stride, 1]
            padding = padding_nhwc
            if data_format == 'NCHW':
                t0 = array_ops.transpose(t0, [0, 3, 1, 2])
                t2 = array_ops.transpose(t2, [0, 3, 1, 2])
                strides = [1, 1, stride, stride]
                padding = padding_nchw
            backprop = nn_ops.depthwise_conv2d_native_backprop_filter(t0, t1, t2, strides=strides, padding=padding, data_format=data_format)
            ret = self.evaluate(backprop)
            self.assertShapeEqual(ret, backprop)
            return ret
    cpu_value = _GetVal(use_gpu=False, dtype=dtype)
    for data_format in ['NHWC', 'NCHW']:
        gpu_value = _GetVal(use_gpu=True, dtype=dtype, data_format=data_format)
        self.assertAllCloseAccordingToType(cpu_value, gpu_value, rtol=0.0001, atol=0.0001, bfloat16_rtol=1.0)