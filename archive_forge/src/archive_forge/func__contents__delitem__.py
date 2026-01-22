from __future__ import annotations
import typing
import warnings
from urwid.canvas import CanvasCombine, CompositeCanvas
from urwid.split_repr import remove_defaults
from urwid.util import is_mouse_press
from .constants import Sizing, VAlign
from .container import WidgetContainerMixin
from .filler import Filler
from .widget import Widget, WidgetError
def _contents__delitem__(self, key: Literal['header', 'footer']) -> None:
    if key not in {'header', 'footer'}:
        raise KeyError(f"Frame.contents can't remove key: {key!r}")
    if key == 'header' and self._header is None or (key == 'footer' and self._footer is None):
        raise KeyError(f'Frame.contents has no key: {key!r}')
    if key == 'header':
        self.header = None
    else:
        self.footer = None