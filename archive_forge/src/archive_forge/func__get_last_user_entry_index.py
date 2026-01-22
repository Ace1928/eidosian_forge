from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from io import BytesIO
from typing import (
import param
from ..io.resources import CDN_DIST
from ..layout import Row, Tabs
from ..pane.image import ImageBase
from ..viewable import Viewable
from ..widgets.base import Widget
from ..widgets.button import Button
from ..widgets.input import FileInput, TextInput
from .feed import CallbackState, ChatFeed
from .input import ChatAreaInput
from .message import ChatMessage, _FileInputMessage
def _get_last_user_entry_index(self) -> int:
    """
        Get the index of the last user message.
        """
    messages = self.objects[::-1]
    for index, message in enumerate(messages, 1):
        if message.user == self.user:
            return index
    return 0