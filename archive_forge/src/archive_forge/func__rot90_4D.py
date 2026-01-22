import functools
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_case
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _rot90_4D(images, k, name_scope):
    """Rotate batch of images counter-clockwise by 90 degrees `k` times.

  Args:
    images: 4-D Tensor of shape `[height, width, channels]`.
    k: A scalar integer. The number of times the images are rotated by 90
      degrees.
    name_scope: A valid TensorFlow name scope.

  Returns:
    A 4-D `Tensor` of the same type and shape as `images`.
  """

    def _rot90():
        return array_ops.transpose(array_ops.reverse_v2(images, [2]), [0, 2, 1, 3])

    def _rot180():
        return array_ops.reverse_v2(images, [1, 2])

    def _rot270():
        return array_ops.reverse_v2(array_ops.transpose(images, [0, 2, 1, 3]), [2])
    cases = [(math_ops.equal(k, 1), _rot90), (math_ops.equal(k, 2), _rot180), (math_ops.equal(k, 3), _rot270)]
    result = control_flow_case.case(cases, default=lambda: images, exclusive=True, name=name_scope)
    shape = result.get_shape()
    result.set_shape([shape[0], None, None, shape[3]])
    return result