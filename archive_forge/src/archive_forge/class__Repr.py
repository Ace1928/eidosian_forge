import inspect
import keyword
import pydoc
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type, Dict, List, ContextManager
from types import MemberDescriptorType, TracebackType
from ._typing_compat import Literal
from pygments.token import Token
from pygments.lexers import Python3Lexer
from .lazyre import LazyReCompile
class _Repr:
    """
    Helper for `ArgSpec`: Returns the given value in `__repr__()`.
    """
    __slots__ = ('value',)

    def __init__(self, value: str) -> None:
        self.value = value

    def __repr__(self) -> str:
        return self.value
    __str__ = __repr__