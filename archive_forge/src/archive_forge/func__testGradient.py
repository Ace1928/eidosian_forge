import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
def _testGradient(self, np_input, bias, dtype, data_format, use_gpu):
    with self.cached_session(use_gpu=use_gpu):
        if data_format == 'NCHW':
            np_input = self._NHWCToNCHW(np_input)
        jacob_a, jacob_n = self._computeGradient(np_input, bias, dtype, data_format)
        input_jacob_a, bias_jacob_a, grad_jacob_a = jacob_a
        input_jacob_n, bias_jacob_n, grad_jacob_n = jacob_n
        if dtype in [np.float16, dtypes.bfloat16.as_numpy_dtype]:
            _, jacob_n = self._computeGradient(np_input, bias, np.float32, data_format)
            input_jacob_n, bias_jacob_n, grad_jacob_n = jacob_n
        if dtype == dtypes.float64:
            threshold = 1e-10
        elif np_input.size >= 512:
            threshold = 0.05
        else:
            threshold = 0.005
        self.assertAllClose(input_jacob_a, input_jacob_n, threshold, threshold)
        self.assertAllClose(bias_jacob_a, bias_jacob_n, threshold, threshold)
        self.assertAllClose(grad_jacob_a, grad_jacob_n, threshold, threshold)