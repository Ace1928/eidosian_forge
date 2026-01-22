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
def _AssertGrayscaleImage(image):
    """Assert that we are working with a properly shaped grayscale image.

  Performs the check statically if possible (i.e. if the shape
  is statically known). Otherwise adds a control dependency
  to an assert op that checks the dynamic shape.

  Args:
    image: >= 2-D Tensor of size [*, 1]

  Raises:
    ValueError: if image.shape is not a [>= 2] vector or if
              last dimension is not size 1.

  Returns:
    If the shape of `image` could be verified statically, `image` is
    returned unchanged, otherwise there will be a control dependency
    added that asserts the correct dynamic shape.
  """
    return control_flow_ops.with_dependencies(_CheckGrayscaleImage(image, require_static=False), image)