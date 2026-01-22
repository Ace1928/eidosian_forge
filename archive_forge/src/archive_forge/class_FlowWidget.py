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
class FlowWidget(Widget):
    """
    Deprecated.  Inherit from Widget and add:

        _sizing = frozenset(['flow'])

    at the top of your class definition instead.

    Base class of widgets that determine their rows from the number of
    columns available.
    """
    _sizing = frozenset([Sizing.FLOW])

    def __init__(self, *args, **kwargs):
        warnings.warn("\n            FlowWidget is deprecated. Inherit from Widget and add:\n\n                _sizing = frozenset(['flow'])\n\n            at the top of your class definition instead.", DeprecationWarning, stacklevel=3)
        super().__init__()

    def rows(self, size: int, focus: bool=False) -> int:
        """
        All flow widgets must implement this function.
        """
        raise NotImplementedError()

    def render(self, size: tuple[int], focus: bool=False) -> Canvas:
        """
        All widgets must implement this function.
        """
        raise NotImplementedError()