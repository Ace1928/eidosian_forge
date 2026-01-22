import itertools
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_inspect
class RegisterCholesky:
    """Decorator to register a Cholesky implementation function.

  Usage:

  @linear_operator_algebra.RegisterCholesky(lin_op.LinearOperatorIdentity)
  def _cholesky_identity(lin_op_a):
    # Return the identity matrix.
  """

    def __init__(self, lin_op_cls_a):
        """Initialize the LinearOperator registrar.

    Args:
      lin_op_cls_a: the class of the LinearOperator to decompose.
    """
        self._key = (lin_op_cls_a,)

    def __call__(self, cholesky_fn):
        """Perform the Cholesky registration.

    Args:
      cholesky_fn: The function to use for the Cholesky.

    Returns:
      cholesky_fn

    Raises:
      TypeError: if cholesky_fn is not a callable.
      ValueError: if a Cholesky function has already been registered for
        the given argument classes.
    """
        if not callable(cholesky_fn):
            raise TypeError('cholesky_fn must be callable, received: {}'.format(cholesky_fn))
        if self._key in _CHOLESKY_DECOMPS:
            raise ValueError('Cholesky({}) has already been registered to: {}'.format(self._key[0].__name__, _CHOLESKY_DECOMPS[self._key]))
        _CHOLESKY_DECOMPS[self._key] = cholesky_fn
        return cholesky_fn