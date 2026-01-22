import functools
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.module import module
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_decorator
def _has_kwargs(fn):
    """Returns whether the passed callable has **kwargs in its signature.

  Args:
    fn: Function, or function-like object (e.g., result of `functools.partial`).

  Returns:
    `bool`: if `fn` has **kwargs in its signature.

  Raises:
     `TypeError`: If fn is not a Function, or function-like object.
  """
    if isinstance(fn, functools.partial):
        fn = fn.func
    elif _is_callable_object(fn):
        fn = fn.__call__
    elif not callable(fn):
        raise TypeError('fn should be a function-like object, but is of type {}.'.format(type(fn)))
    return tf_inspect.getfullargspec(fn).varkw is not None