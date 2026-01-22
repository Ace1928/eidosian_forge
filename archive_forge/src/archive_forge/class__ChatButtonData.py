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
@dataclass
class _ChatButtonData:
    """
    A dataclass to hold the metadata and data related to the
    chat buttons.

    Parameters
    ----------
    index : int
        The index of the button.
    name : str
        The name of the button.
    icon : str
        The icon to display.
    objects : List
        The objects to display.
    buttons : List
        The buttons to display.
    """
    index: int
    name: str
    icon: str
    objects: List
    buttons: List
    callback: Callable