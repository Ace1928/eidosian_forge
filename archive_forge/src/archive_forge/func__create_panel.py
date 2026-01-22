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
def _create_panel(self, value, old=None):
    """
        Create a panel object from the value.
        """
    if isinstance(value, Viewable):
        self._internal = False
        self._include_stylesheets_inplace(value)
        self._include_message_css_class_inplace(value)
        return value
    renderer = None
    if isinstance(value, _FileInputMessage):
        contents = value.contents
        mime_type = value.mime_type
        value, renderer = self._select_renderer(contents, mime_type)
    else:
        try:
            import magic
            mime_type = magic.from_buffer(value, mime=True)
            value, renderer = self._select_renderer(value, mime_type)
        except Exception:
            pass
    renderers = self.renderers.copy() or []
    if renderer is not None:
        renderers.append(renderer)
    for renderer in renderers:
        try:
            if self._is_widget_renderer(renderer):
                object_panel = renderer(value=value)
            else:
                object_panel = renderer(value)
            if isinstance(object_panel, Viewable):
                break
        except Exception:
            pass
    else:
        if isinstance(old, Markdown) and isinstance(value, str):
            self._set_params(old, object=value)
            return old
        object_panel = _panel(value)
    self._set_params(object_panel)
    if type(old) is type(object_panel) and self._internal:
        ReplacementPane._recursive_update(old, object_panel)
        return object_panel
    self._internal = True
    return object_panel