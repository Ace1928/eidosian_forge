from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import math_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
Perform the KL registration.

    Args:
      kl_fn: The function to use for the KL divergence.

    Returns:
      kl_fn

    Raises:
      TypeError: if kl_fn is not a callable.
      ValueError: if a KL divergence function has already been registered for
        the given argument classes.
    