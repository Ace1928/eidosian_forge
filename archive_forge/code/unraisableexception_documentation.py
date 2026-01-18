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
Context manager catching unraisable exception using sys.unraisablehook.

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
    