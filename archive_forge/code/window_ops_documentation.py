import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
Helper function for computing a raised cosine window.

  Args:
    name: Name to use for the scope.
    default_name: Default name to use for the scope.
    window_length: A scalar `Tensor` or integer indicating the window length.
    periodic: A bool `Tensor` indicating whether to generate a periodic or
      symmetric window.
    dtype: A floating point `DType`.
    a: The alpha parameter to the raised cosine window.
    b: The beta parameter to the raised cosine window.

  Returns:
    A `Tensor` of shape `[window_length]` of type `dtype`.

  Raises:
    ValueError: If `dtype` is not a floating point type or `window_length` is
      not scalar or `periodic` is not scalar.
  