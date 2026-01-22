from __future__ import annotations
import typing as t
from base64 import b64decode
from base64 import b64encode
from datetime import datetime
from uuid import UUID
from markupsafe import Markup
from werkzeug.http import http_date
from werkzeug.http import parse_date
from ..json import dumps
from ..json import loads
def _untag_scan(self, value: t.Any) -> t.Any:
    if isinstance(value, dict):
        value = {k: self._untag_scan(v) for k, v in value.items()}
        value = self.untag(value)
    elif isinstance(value, list):
        value = [self._untag_scan(item) for item in value]
    return value