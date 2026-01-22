import contextlib
import io
import os
from uuid import uuid4
import requests
from .._compat import fields
def _next_part(self):
    try:
        p = self._current_part = next(self._iter_parts)
    except StopIteration:
        p = None
    return p