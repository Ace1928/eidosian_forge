from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
import sys
import types
from fire import docstrings
import six
def _GetArgSpecInfo(fn):
    """Gives information pertaining to computing the ArgSpec of fn.

  Determines if the first arg is supplied automatically when fn is called.
  This arg will be supplied automatically if fn is a bound method or a class
  with an __init__ method.

  Also returns the function who's ArgSpec should be used for determining the
  calling parameters for fn. This may be different from fn itself if fn is a
  class with an __init__ method.

  Args:
    fn: The function or class of interest.
  Returns:
    A tuple with the following two items:
      fn: The function to use for determining the arg spec of this function.
      skip_arg: Whether the first argument will be supplied automatically, and
        hence should be skipped when supplying args from a Fire command.
  """
    skip_arg = False
    if inspect.isclass(fn):
        skip_arg = True
        if six.PY2 and hasattr(fn, '__init__'):
            fn = fn.__init__
    elif inspect.ismethod(fn):
        skip_arg = fn.__self__ is not None
    elif inspect.isbuiltin(fn):
        if not isinstance(fn.__self__, types.ModuleType):
            skip_arg = True
    elif not inspect.isfunction(fn):
        skip_arg = True
    return (fn, skip_arg)