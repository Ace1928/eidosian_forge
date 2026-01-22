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
def _render_avatar(self) -> HTML | Image:
    """
        Render the avatar pane as some HTML text or Image pane.
        """
    avatar = self.avatar
    if not avatar and self.user:
        avatar = self.user[0]
    avatar_params = {'css_classes': ['avatar']}
    if isinstance(avatar, ImageBase):
        avatar_pane = avatar
        avatar_params['css_classes'] = avatar_params.get('css_classes', []) + avatar_pane.css_classes
        avatar_params.update(width=35, height=35)
        avatar_pane.param.update(avatar_params)
    elif not isinstance(avatar, (BytesIO, bytes)) and len(avatar) == 1:
        avatar_pane = HTML(avatar, **avatar_params)
    else:
        try:
            avatar_pane = Image(avatar, width=35, height=35, **avatar_params)
        except ValueError:
            avatar_pane = HTML(avatar, **avatar_params)
    return avatar_pane