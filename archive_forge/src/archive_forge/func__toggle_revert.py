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
def _toggle_revert(self, button_data: _ChatButtonData, active: bool):
    """
        Toggle the button's icon and name to indicate
        whether the action can be reverted.
        """
    for button in button_data.buttons:
        if active and button_data.objects:
            button_update = {'button_type': 'warning', 'name': 'Revert', 'width': 90}
        else:
            button_update = {'button_type': 'default', 'name': button_data.name.title() if self.show_button_name else '', 'width': 90 if self.show_button_name else 45}
        button.param.update(button_update)