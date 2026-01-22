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
def _CheckGrayscaleImage(image, require_static=True):
    """Assert that we are working with properly shaped grayscale image.

  Args:
    image: >= 2-D Tensor of size [*, 1]
    require_static: Boolean, whether static shape is required.

  Raises:
    ValueError: if image.shape is not a [>= 2] vector or if
              last dimension is not size 1.

  Returns:
    An empty list, if `image` has fully defined dimensions. Otherwise, a list
    containing an assert op is returned.
  """
    try:
        if image.get_shape().ndims is None:
            image_shape = image.get_shape().with_rank(2)
        else:
            image_shape = image.get_shape().with_rank_at_least(2)
    except ValueError:
        raise ValueError('A grayscale image (shape %s) must be at least two-dimensional.' % image.shape)
    if require_static and (not image_shape.is_fully_defined()):
        raise ValueError("'image' must be fully defined.")
    if image_shape.is_fully_defined():
        if image_shape[-1] != 1:
            raise ValueError('Last dimension of a grayscale image should be size 1.')
    if not image_shape.is_fully_defined():
        return [check_ops.assert_equal(array_ops.shape(image)[-1], 1, message='Last dimension of a grayscale image should be size 1.'), check_ops.assert_greater_equal(array_ops.rank(image), 3, message='A grayscale image must be at least two-dimensional.')]
    else:
        return []