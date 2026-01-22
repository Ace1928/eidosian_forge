from __future__ import annotations
import re
import typing as t
from .._internal import _missing
from ..exceptions import BadRequestKeyError
from .mixins import ImmutableHeadersMixin
from .structures import iter_multi_items
from .structures import MultiDict
from .. import http
def _options_header_vkw(value: str, kw: dict[str, t.Any]):
    return http.dump_options_header(value, {k.replace('_', '-'): v for k, v in kw.items()})