import collections
import functools
import inspect
import re
from tensorflow.python.framework import strict_mode
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import decorator_utils
from tensorflow.python.util import is_in_graph_mode
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.tools.docs import doc_controls
def _wrap_decorator(wrapped_function, decorator_name):
    """Indicate that one function wraps another.

  This decorator wraps a function using `tf_decorator.make_decorator`
  so that doc generation scripts can pick up original function
  signature.
  It would be better to use @functools.wrap decorator, but it would
  not update function signature to match wrapped function in Python 2.

  Args:
    wrapped_function: The function that decorated function wraps.
    decorator_name: The name of the decorator.

  Returns:
    Function that accepts wrapper function as an argument and returns
    `TFDecorator` instance.
  """

    def wrapper(wrapper_func):
        return tf_decorator.make_decorator(wrapped_function, wrapper_func, decorator_name)
    return wrapper