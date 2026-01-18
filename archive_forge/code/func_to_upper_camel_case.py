from __future__ import annotations
import re
from typing import Any, Callable
def to_upper_camel_case(snake_case_str: str) -> str:
    """Converts snake_case to UpperCamelCase.

    Example
    -------
        foo_bar -> FooBar

    """
    return ''.join(map(str.title, snake_case_str.split('_')))