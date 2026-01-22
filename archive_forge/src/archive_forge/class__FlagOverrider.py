import collections
import functools
import inspect
from typing import overload, Any, Callable, Mapping, Tuple, TypeVar, Type, Sequence, Union
from absl import flags
class _FlagOverrider(object):
    """Overrides flags for the duration of the decorated function call.

  It also restores all original values of flags after decorated method
  completes.
  """

    def __init__(self, **overrides: Any):
        self._overrides = overrides
        self._saved_flag_values = None

    def __call__(self, func: _CallableT) -> _CallableT:
        if inspect.isclass(func):
            raise TypeError('flagsaver cannot be applied to a class.')
        return _wrap(self.__class__, func, self._overrides)

    def __enter__(self):
        self._saved_flag_values = save_flag_values(FLAGS)
        try:
            FLAGS._set_attributes(**self._overrides)
        except:
            restore_flag_values(self._saved_flag_values, FLAGS)
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        restore_flag_values(self._saved_flag_values, FLAGS)