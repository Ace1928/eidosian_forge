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
def _add_or_replace_parameters(url: str, params: Dict[str, str]) -> str:
    parsed = urlsplit(url)
    current_args = parse_qsl(parsed.query, keep_blank_values=True)
    new_args = []
    seen_params = set()
    for name, value in current_args:
        if name not in params:
            new_args.append((name, value))
        elif name not in seen_params:
            new_args.append((name, params[name]))
            seen_params.add(name)
    not_modified_args = [(name, value) for name, value in params.items() if name not in seen_params]
    new_args += not_modified_args
    query = urlencode(new_args)
    return urlunsplit(parsed._replace(query=query))