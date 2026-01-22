from __future__ import annotations
import datetime
import re
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from io import BytesIO
from tempfile import NamedTemporaryFile
from textwrap import indent
from typing import (
from zoneinfo import ZoneInfo
import param
from ..io.resources import CDN_DIST, get_dist_path
from ..io.state import state
from ..layout import Column, Row
from ..pane.base import PaneBase, ReplacementPane, panel as _panel
from ..pane.image import (
from ..pane.markup import (
from ..pane.media import Audio, Video
from ..param import ParamFunction
from ..viewable import Viewable
from ..widgets.base import Widget
from .icon import ChatCopyIcon, ChatReactionIcons
def _update_chat_copy_icon(self):
    object_panel = self._object_panel
    if isinstance(object_panel, HTMLBasePane):
        object_panel = object_panel.object
    elif isinstance(object_panel, Widget):
        object_panel = object_panel.value
    if isinstance(object_panel, str) and self.show_copy_icon:
        self.chat_copy_icon.value = object_panel
        self.chat_copy_icon.visible = True
    else:
        self.chat_copy_icon.value = ''
        self.chat_copy_icon.visible = False