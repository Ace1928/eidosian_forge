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
def _select_renderer(self, contents: Any, mime_type: str):
    """
        Determine the renderer to use based on the mime type.
        """
    renderer = _panel
    if mime_type == 'application/pdf':
        contents = self._exit_stack.enter_context(BytesIO(contents))
        renderer = partial(PDF, embed=True)
    elif mime_type.startswith('audio/'):
        file = self._exit_stack.enter_context(NamedTemporaryFile(suffix='.mp3', delete=False))
        file.write(contents)
        file.seek(0)
        contents = file.name
        renderer = Audio
    elif mime_type.startswith('video/'):
        contents = self._exit_stack.enter_context(BytesIO(contents))
        renderer = Video
    elif mime_type.startswith('image/'):
        contents = self._exit_stack.enter_context(BytesIO(contents))
        renderer = Image
    elif mime_type.endswith('/csv'):
        import pandas as pd
        with BytesIO(contents) as buf:
            contents = pd.read_csv(buf)
        renderer = DataFrame
    elif mime_type.startswith('text'):
        if isinstance(contents, bytes):
            contents = contents.decode('utf-8')
    return (contents, renderer)