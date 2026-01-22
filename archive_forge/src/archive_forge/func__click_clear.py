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
def _click_clear(self, event: param.parameterized.Event | None=None, instance: 'ChatInterface' | None=None) -> None:
    """
        Upon clicking the clear button, clear the chat log.
        If the button is clicked again without performing any
        other actions, revert the clear.
        """
    clear_data = self._button_data['clear']
    clear_objects = clear_data.objects
    if not clear_objects:
        self._reset_button_data()
        clear_data.objects = self.clear()
        if self._allow_revert:
            self._toggle_revert(clear_data, True)
        else:
            clear_data.objects = []
    else:
        self[:] = clear_objects.copy()
        self._reset_button_data()