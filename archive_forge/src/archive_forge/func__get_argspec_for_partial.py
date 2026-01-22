import collections
import functools
import inspect as _inspect
from tensorflow.python.util import tf_decorator
def _get_argspec_for_partial(obj):
    """Implements `getargspec` for `functools.partial` objects.

  Args:
    obj: The `functools.partial` object
  Returns:
    An `inspect.ArgSpec`
  Raises:
    ValueError: When callable's signature can not be expressed with
      ArgSpec.
  """
    n_prune_args = len(obj.args)
    partial_keywords = obj.keywords or {}
    args, varargs, keywords, defaults = getargspec(obj.func)
    args = args[n_prune_args:]
    no_default = object()
    all_defaults = [no_default] * len(args)
    if defaults:
        all_defaults[-len(defaults):] = defaults
    for kw, default in partial_keywords.items():
        if kw in args:
            idx = args.index(kw)
            all_defaults[idx] = default
        elif not keywords:
            raise ValueError('Function does not have **kwargs parameter, but contains an unknown partial keyword.')
    first_default = next((idx for idx, x in enumerate(all_defaults) if x is not no_default), None)
    if first_default is None:
        return ArgSpec(args, varargs, keywords, None)
    invalid_default_values = [args[i] for i, j in enumerate(all_defaults) if j is no_default and i > first_default]
    if invalid_default_values:
        raise ValueError('Some arguments %s do not have default value, but they are positioned after those with default values. This can not be expressed with ArgSpec.' % invalid_default_values)
    return ArgSpec(args, varargs, keywords, tuple(all_defaults[first_default:]))