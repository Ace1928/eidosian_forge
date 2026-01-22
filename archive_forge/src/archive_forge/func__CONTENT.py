from __future__ import annotations
import logging # isort:skip
import json
from typing import (
from .exceptions import ValidationError
from .message import BufferHeader, Message
def _CONTENT(self, fragment: Fragment) -> None:
    content = self._assume_text(fragment)
    self._fragments.append(content)
    header_json, metadata_json, content_json = (self._assume_text(x) for x in self._fragments[:3])
    self._partial = self._protocol.assemble(header_json, metadata_json, content_json)
    self._check_complete()