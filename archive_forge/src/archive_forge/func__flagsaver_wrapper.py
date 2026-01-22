import collections
import functools
import inspect
from typing import overload, Any, Callable, Mapping, Tuple, TypeVar, Type, Sequence, Union
from absl import flags
@functools.wraps(func)
def _flagsaver_wrapper(*args, **kwargs):
    """Wrapper function that saves and restores flags."""
    with flag_overrider_cls(**overrides):
        return func(*args, **kwargs)