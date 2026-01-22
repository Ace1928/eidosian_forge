import numpy as np
from tensorflow.python.eager import backprop as backprop_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
@test_util.run_in_graph_and_eager_modes(use_gpu=False)
@test_util.disable_xla('XLA cannot assert inside of a kernel.')
def _testInvalidLabelCPU(self, expected_regex='Received a label value of'):
    labels = [4, 3, 0, -1]
    logits = [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]
    with self.assertRaisesRegex((errors_impl.InvalidArgumentError, errors_impl.UnknownError), expected_regex):
        self.evaluate(nn_ops.sparse_softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))