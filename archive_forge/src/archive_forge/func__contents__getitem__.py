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
def _contents__getitem__(self, key: Literal['body', 'header', 'footer']) -> tuple[BodyWidget | HeaderWidget | FooterWidget, None]:
    if key == 'body':
        return (self._body, None)
    if key == 'header' and self._header:
        return (self._header, None)
    if key == 'footer' and self._footer:
        return (self._footer, None)
    raise KeyError(f'Frame.contents has no key: {key!r}')