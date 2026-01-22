from __future__ import annotations
import asyncio
import traceback
from enum import Enum
from inspect import (
from io import BytesIO
from typing import (
import param
from .._param import Margin
from ..io.resources import CDN_DIST
from ..layout import Feed, ListPanel
from ..layout.card import Card
from ..layout.spacer import VSpacer
from ..pane.image import SVG
from .message import ChatMessage
@param.depends('load_buffer', 'auto_scroll_limit', 'scroll_button_threshold', watch=True)
def _update_chat_log_params(self):
    self._chat_log.load_buffer = self.load_buffer
    self._chat_log.auto_scroll_limit = self.auto_scroll_limit
    self._chat_log.scroll_button_threshold = self.scroll_button_threshold