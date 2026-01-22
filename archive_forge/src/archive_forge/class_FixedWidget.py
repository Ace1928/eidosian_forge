from __future__ import annotations
import functools
import logging
import typing
import warnings
from operator import attrgetter
from urwid import signals
from urwid.canvas import Canvas, CanvasCache, CompositeCanvas
from urwid.command_map import command_map
from urwid.split_repr import split_repr
from urwid.util import MetaSuper
from .constants import Sizing
class FixedWidget(Widget):
    """
    Deprecated.  Inherit from Widget and add:

        _sizing = frozenset(['fixed'])

    at the top of your class definition instead.

    Base class of widgets that know their width and height and
    cannot be resized
    """
    _sizing = frozenset([Sizing.FIXED])

    def __init__(self, *args, **kwargs):
        warnings.warn("\n            FixedWidget is deprecated. Inherit from Widget and add:\n\n                _sizing = frozenset(['fixed'])\n\n            at the top of your class definition instead.", DeprecationWarning, stacklevel=3)
        super().__init__()

    def render(self, size: tuple[()], focus: bool=False) -> Canvas:
        """
        All widgets must implement this function.
        """
        raise NotImplementedError()

    def pack(self, size: tuple[()]=(), focus: bool=False) -> tuple[int, int]:
        """
        All fixed widgets must implement this function.
        """
        raise NotImplementedError()