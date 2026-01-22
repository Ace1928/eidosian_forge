from io import BytesIO
from typing import Callable, Dict, List, Tuple
from .. import errors, osutils, registry
def _reset_registry(value=None):
    """Reset the filter stack registry.

    This function is provided to aid testing. The expected usage is::

      old = _reset_registry()
      # run tests
      _reset_registry(old)

    Args:
      value: the value to set the registry to or None for an empty one.

    Returns:
      the existing value before it reset.
    """
    global filter_stacks_registry
    original = filter_stacks_registry
    if value is None:
        filter_stacks_registry = registry.Registry()
    else:
        filter_stacks_registry = value
    _stack_cache.clear()
    return original