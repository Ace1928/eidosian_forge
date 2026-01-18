import itertools
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_inspect
Perform the Inverse registration.

    Args:
      inverse_fn: The function to use for the Inverse.

    Returns:
      inverse_fn

    Raises:
      TypeError: if inverse_fn is not a callable.
      ValueError: if a Inverse function has already been registered for
        the given argument classes.
    