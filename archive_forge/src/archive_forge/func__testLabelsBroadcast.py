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
@test_util.run_deprecated_v1
def _testLabelsBroadcast(self, uniform_labels_gradient):
    labels = np.array([[0.0, 0.0, 0.0, 1.0]]).astype(np.float16)
    logits = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]]).astype(np.float16)
    self._testXent2D(labels, logits, with_placeholders=True)
    labels = np.array([[1.0]]).astype(np.float16)
    logits = np.array([[1.0], [2.0]]).astype(np.float16)
    self._testXent2D(labels, logits, with_placeholders=True)
    labels = np.array([[0.0], [2.0], [0.25]]).astype(np.float16)
    logits = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]).astype(np.float16)
    self._testXent2D(labels, logits, with_placeholders=True, expected_gradient=uniform_labels_gradient)