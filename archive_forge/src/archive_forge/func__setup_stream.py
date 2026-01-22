from __future__ import annotations
import functools
import json
import textwrap
from typing import (
import param  # type: ignore
from ..io.resources import CDN_DIST
from ..models import HTML as _BkHTML, JSON as _BkJSON
from ..util import HTML_SANITIZER, escape
from .base import ModelPane
@param.depends('object', watch=True, on_init=True)
def _setup_stream(self):
    if not self._models or not hasattr(self.object, 'stream'):
        return
    elif self._stream:
        self._stream.destroy()
        self._stream = None
    self._stream = self.object.stream.latest().rate_limit(0.5).gather()
    self._stream.sink(self._set_object)