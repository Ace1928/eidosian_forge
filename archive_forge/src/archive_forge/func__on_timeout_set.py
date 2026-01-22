from __future__ import annotations
from vine import transform
from .message import AsyncMessage
def _on_timeout_set(self, visibility_timeout):
    if visibility_timeout:
        self.visibility_timeout = visibility_timeout
    return self.visibility_timeout