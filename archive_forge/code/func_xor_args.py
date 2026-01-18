import contextlib
import datetime
import functools
import importlib
import warnings
from importlib.metadata import version
from typing import Any, Callable, Dict, Optional, Set, Tuple, Union
from packaging.version import parse
from requests import HTTPError, Response
from langchain_core.pydantic_v1 import SecretStr
def xor_args(*arg_groups: Tuple[str, ...]) -> Callable:
    """Validate specified keyword args are mutually exclusive."""

    def decorator(func: Callable) -> Callable:

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Validate exactly one arg in each group is not None."""
            counts = [sum((1 for arg in arg_group if kwargs.get(arg) is not None)) for arg_group in arg_groups]
            invalid_groups = [i for i, count in enumerate(counts) if count != 1]
            if invalid_groups:
                invalid_group_names = [', '.join(arg_groups[i]) for i in invalid_groups]
                raise ValueError(f'Exactly one argument in each of the following groups must be defined: {', '.join(invalid_group_names)}')
            return func(*args, **kwargs)
        return wrapper
    return decorator