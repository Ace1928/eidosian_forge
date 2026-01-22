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
def _serialize_recursively(self, obj: Any, prefix_with_viewable_label: bool=True, prefix_with_container_label: bool=True) -> str:
    """
        Recursively serialize the object to a string.
        """
    if isinstance(obj, Iterable) and (not isinstance(obj, str)):
        content = tuple((self._serialize_recursively(o, prefix_with_viewable_label=prefix_with_viewable_label, prefix_with_container_label=prefix_with_container_label) for o in obj))
        if prefix_with_container_label:
            if len(content) == 1:
                return f'{self._get_obj_label(obj)}({content[0]})'
            else:
                indented_content = indent(',\n'.join(content), prefix=' ' * 4)
                return f'{self._get_obj_label(obj)}(\n{indented_content}\n)'
        else:
            return f'({', '.join(content)})'
    string = obj
    if hasattr(obj, 'value'):
        string = obj.value
    elif hasattr(obj, 'object'):
        string = obj.object
    if hasattr(string, 'decode') or isinstance(string, BytesIO):
        self.param.warning(f'Serializing byte-like objects are not supported yet; using the label of the object as a placeholder for {obj}')
        return self._get_obj_label(obj)
    if prefix_with_viewable_label and isinstance(obj, Viewable):
        label = self._get_obj_label(obj)
        string = f'{label}={string!r}'
    return string