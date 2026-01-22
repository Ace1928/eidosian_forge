import sys
import traceback
from types import TracebackType
from typing import Any
from typing import Callable
from typing import Generator
from typing import Optional
from typing import Type
import warnings
import pytest
class catch_unraisable_exception:
    """Context manager catching unraisable exception using sys.unraisablehook.

    Storing the exception value (cm.unraisable.exc_value) creates a reference
    cycle. The reference cycle is broken explicitly when the context manager
    exits.

    Storing the object (cm.unraisable.object) can resurrect it if it is set to
    an object which is being finalized. Exiting the context manager clears the
    stored object.

    Usage:
        with catch_unraisable_exception() as cm:
            # code creating an "unraisable exception"
            ...
            # check the unraisable exception: use cm.unraisable
            ...
        # cm.unraisable attribute no longer exists at this point
        # (to break a reference cycle)
    """

    def __init__(self) -> None:
        self.unraisable: Optional['sys.UnraisableHookArgs'] = None
        self._old_hook: Optional[Callable[['sys.UnraisableHookArgs'], Any]] = None

    def _hook(self, unraisable: 'sys.UnraisableHookArgs') -> None:
        self.unraisable = unraisable

    def __enter__(self) -> 'catch_unraisable_exception':
        self._old_hook = sys.unraisablehook
        sys.unraisablehook = self._hook
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> None:
        assert self._old_hook is not None
        sys.unraisablehook = self._old_hook
        self._old_hook = None
        del self.unraisable