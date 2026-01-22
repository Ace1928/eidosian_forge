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
def _wrap_callbacks(self, callback: Callable | None=None, post_callback: Callable | None=None, name: str=''):
    """
        Wrap the callback and post callback around the default callback.
        """

    def decorate(default_callback: Callable):

        def wrapper(self, event: param.parameterized.Event):
            if name == 'send' and (not self.active_widget.value):
                return
            if callback is not None:
                try:
                    self.disabled = True
                    callback(self, event)
                finally:
                    self.disabled = False
            default_callback(self, event)
            if post_callback is not None:
                try:
                    self.disabled = True
                    post_callback(self, event)
                finally:
                    self.disabled = False
        return wrapper
    return decorate