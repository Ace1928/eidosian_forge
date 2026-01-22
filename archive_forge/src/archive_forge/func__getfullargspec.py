import collections
import functools
import inspect as _inspect
from tensorflow.python.util import tf_decorator
def _getfullargspec(target):
    """A python2 version of getfullargspec.

    Args:
      target: the target object to inspect.

    Returns:
      A FullArgSpec with empty kwonlyargs, kwonlydefaults and annotations.
    """
    return _convert_maybe_argspec_to_fullargspec(getargspec(target))