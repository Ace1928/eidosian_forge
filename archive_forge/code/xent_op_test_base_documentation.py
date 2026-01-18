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
This method is structured to be easily overridden by a child class.