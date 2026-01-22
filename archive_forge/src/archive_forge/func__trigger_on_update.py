from __future__ import annotations
import base64
import binascii
import typing as t
from ..http import dump_header
from ..http import parse_dict_header
from ..http import quote_header_value
from .structures import CallbackDict
def _trigger_on_update(self) -> None:
    if self._on_update is not None:
        self._on_update(self)