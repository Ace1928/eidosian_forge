import itertools
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_inspect
class RegisterSolve:
    """Decorator to register a Solve implementation function.

  Usage:

  @linear_operator_algebra.RegisterSolve(
    lin_op.LinearOperatorIdentity,
    lin_op.LinearOperatorIdentity)
  def _solve_identity(a, b):
    # Return the identity matrix.
  """

    def __init__(self, lin_op_cls_a, lin_op_cls_b):
        """Initialize the LinearOperator registrar.

    Args:
      lin_op_cls_a: the class of the LinearOperator that is computing solve.
      lin_op_cls_b: the class of the second LinearOperator to solve.
    """
        self._key = (lin_op_cls_a, lin_op_cls_b)

    def __call__(self, solve_fn):
        """Perform the Solve registration.

    Args:
      solve_fn: The function to use for the Solve.

    Returns:
      solve_fn

    Raises:
      TypeError: if solve_fn is not a callable.
      ValueError: if a Solve function has already been registered for
        the given argument classes.
    """
        if not callable(solve_fn):
            raise TypeError('solve_fn must be callable, received: {}'.format(solve_fn))
        if self._key in _SOLVE:
            raise ValueError('Solve({}, {}) has already been registered.'.format(self._key[0].__name__, self._key[1].__name__))
        _SOLVE[self._key] = solve_fn
        return solve_fn