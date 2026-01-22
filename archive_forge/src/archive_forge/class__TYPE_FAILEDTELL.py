from __future__ import annotations
import io
import typing
from base64 import b64encode
from enum import Enum
from ..exceptions import UnrewindableBodyError
from .util import to_bytes
class _TYPE_FAILEDTELL(Enum):
    token = 0