from __future__ import annotations
import re
import typing as t
from .._internal import _missing
from ..exceptions import BadRequestKeyError
from .mixins import ImmutableHeadersMixin
from .structures import iter_multi_items
from .structures import MultiDict
from .. import http
def _str_header_value(value: t.Any) -> str:
    if not isinstance(value, str):
        value = str(value)
    if _newline_re.search(value) is not None:
        raise ValueError('Header values must not contain newline characters.')
    return value