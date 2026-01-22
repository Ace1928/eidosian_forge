from __future__ import annotations
import logging # isort:skip
import base64
import calendar
import codecs
import datetime as dt
import hashlib
import hmac
import json
import time
import zlib
from typing import TYPE_CHECKING, Any
from ..core.types import ID
from ..settings import settings
from .warnings import warn
class _BytesEncoder(json.JSONEncoder):

    def default(self, o: Any) -> Any:
        if isinstance(o, bytes):
            return dict(bytes=_base64_encode(o))
        return super().default(o)