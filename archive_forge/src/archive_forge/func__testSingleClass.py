import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.nn_grad import _SoftmaxCrossEntropyWithLogitsGrad  # pylint: disable=unused-import
from tensorflow.python.platform import test
def _testSingleClass(self, expected_gradient=[[2.0], [1.0], [0.0], [0.0]]):
    for dtype in (np.float16, np.float32, dtypes.bfloat16.as_numpy_dtype):
        loss, gradient = self._opFwdBwd(labels=np.array([[-1.0], [0.0], [1.0], [1.0]]).astype(dtype), logits=np.array([[1.0], [-1.0], [0.0], [1.0]]).astype(dtype))
        self.assertAllClose([0.0, 0.0, 0.0, 0.0], loss)
        self.assertAllClose(expected_gradient, gradient)