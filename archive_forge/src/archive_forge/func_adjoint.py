import itertools
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_inspect
def adjoint(lin_op_a, name=None):
    """Get the adjoint associated to lin_op_a.

  Args:
    lin_op_a: The LinearOperator to take the adjoint of.
    name: Name to use for this operation.

  Returns:
    A LinearOperator that represents the adjoint of `lin_op_a`.

  Raises:
    NotImplementedError: If no Adjoint method is defined for the LinearOperator
      type of `lin_op_a`.
  """
    adjoint_fn = _registered_adjoint(type(lin_op_a))
    if adjoint_fn is None:
        raise ValueError('No adjoint registered for {}'.format(type(lin_op_a)))
    with ops.name_scope(name, 'Adjoint'):
        return adjoint_fn(lin_op_a)