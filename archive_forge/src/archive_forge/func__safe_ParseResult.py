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
def _safe_ParseResult(parts: ParseResult, encoding: str='utf8', path_encoding: str='utf8') -> Tuple[str, str, str, str, str, str]:
    try:
        netloc = parts.netloc.encode('idna').decode()
    except UnicodeError:
        netloc = parts.netloc
    return (parts.scheme, netloc, quote(parts.path.encode(path_encoding), _path_safe_chars), quote(parts.params.encode(path_encoding), _safe_chars), quote(parts.query.encode(encoding), _safe_chars), quote(parts.fragment.encode(encoding), _safe_chars))