import contextlib
import re
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
def _recursive_apply(tensors, apply_fn):
    """Helper method to recursively apply a function to structure of tensors.

  The structure of the tensors should take the form similar to fetches in
  `tf.compat.v1.Session` and includes single `Tensor`, `list`, nested `list`,
  `tuple`,
  `namedtuple`, or `dict`.

  Args:
    tensors: Single `Tensor`, `list`, nested `list, `tuple`, `namedtuple`, or
      `dict`.
    apply_fn: Function to apply to each `Tensor` and should return a `Tensor`.

  Returns:
    Returns the modified tensors with the same structure.
  Raises:
    `TypeError` if undefined type in the tensors structure.
  """
    tensors_type = type(tensors)
    if isinstance(tensors, tensor_lib.Tensor):
        return apply_fn(tensors)
    elif isinstance(tensors, variables.Variable):
        return apply_fn(tensors.value())
    elif isinstance(tensors, (list, tuple)):
        tensors = [_recursive_apply(t, apply_fn) for t in tensors]
        if tensors_type is list:
            return list(tensors)
        elif tensors_type is tuple:
            return tuple(tensors)
        return tensors_type(*tensors)
    elif tensors_type is dict:
        return dict(((k, _recursive_apply(v, apply_fn)) for k, v in tensors.items()))
    else:
        raise TypeError(f'_recursive_apply argument {tensors!r} has invalid type {tensors_type!r}')