import base64
import codecs
import os
import posixpath
import re
import string
from typing import (
from urllib.parse import (
from urllib.parse import _coerce_args  # type: ignore
from urllib.request import pathname2url, url2pathname
from .util import to_unicode
from ._infra import _ASCII_TAB_OR_NEWLINE, _C0_CONTROL_OR_SPACE
from ._types import AnyUnicodeError, StrOrBytes
from ._url import _SPECIAL_SCHEMES
def add_or_replace_parameter(url: str, name: str, new_value: str) -> str:
    """Add or remove a parameter to a given url

    >>> import w3lib.url
    >>> w3lib.url.add_or_replace_parameter('http://www.example.com/index.php', 'arg', 'v')
    'http://www.example.com/index.php?arg=v'
    >>> w3lib.url.add_or_replace_parameter('http://www.example.com/index.php?arg1=v1&arg2=v2&arg3=v3', 'arg4', 'v4')
    'http://www.example.com/index.php?arg1=v1&arg2=v2&arg3=v3&arg4=v4'
    >>> w3lib.url.add_or_replace_parameter('http://www.example.com/index.php?arg1=v1&arg2=v2&arg3=v3', 'arg3', 'v3new')
    'http://www.example.com/index.php?arg1=v1&arg2=v2&arg3=v3new'
    >>>

    """
    return _add_or_replace_parameters(url, {name: new_value})