from __future__ import annotations
from typing import (
from ..chat.feed import ChatFeed
from ..chat.interface import ChatInterface
from ..chat.message import DEFAULT_AVATARS
from ..layout import Accordion
def _reset_active(self):
    self._active_user = self._input_user
    self._active_avatar = self._input_avatar
    self._message = None