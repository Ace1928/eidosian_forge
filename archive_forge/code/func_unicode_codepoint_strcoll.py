import locale
import threading
from contextlib import AbstractContextManager
from types import TracebackType
from typing import TYPE_CHECKING, Any, Optional, Tuple, Type, Union
from urllib.parse import urljoin, urlsplit
from .exceptions import xpath_error
def unicode_codepoint_strcoll(s1: str, s2: str) -> int:
    return 0 if s1 == s2 else -1 if s1 < s2 else 1