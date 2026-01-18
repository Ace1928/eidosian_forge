import contextlib
import functools
import inspect
import warnings
from typing import Any, Callable, Generator, Type, TypeVar, Union, cast
from langchain_core._api.internal import is_caller_internal
def warning_emitting_wrapper(*args: Any, **kwargs: Any) -> Any:
    """Wrapper for the original wrapped callable that emits a warning.

            Args:
                *args: The positional arguments to the function.
                **kwargs: The keyword arguments to the function.

            Returns:
                The return value of the function being wrapped.
            """
    nonlocal warned
    if not warned and (not is_caller_internal()):
        warned = True
        emit_warning()
    return wrapped(*args, **kwargs)