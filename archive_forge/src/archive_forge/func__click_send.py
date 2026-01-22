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
def _click_send(self, event: param.parameterized.Event | None=None, instance: 'ChatInterface' | None=None) -> None:
    """
        Send the input when the user presses Enter.
        """
    if self.disabled:
        return
    active_widget = self.active_widget
    value = active_widget.value
    if not value and hasattr(active_widget, 'value_input'):
        value = active_widget.value_input
    if value:
        if isinstance(active_widget, FileInput):
            value = _FileInputMessage(contents=value, mime_type=active_widget.mime_type, file_name=active_widget.filename)
        if type(active_widget) is TextInput or self.reset_on_send:
            updates = {'value': ''}
            if hasattr(active_widget, 'value_input'):
                updates['value_input'] = ''
            try:
                with param.discard_events(self):
                    active_widget.param.update(updates)
            except ValueError:
                pass
    else:
        return
    self._reset_button_data()
    self.send(value=value, user=self.user, avatar=self.avatar, respond=True)