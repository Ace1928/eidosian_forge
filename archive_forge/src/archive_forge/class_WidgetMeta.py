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
class WidgetMeta(MetaSuper, signals.MetaSignals):
    """
    Bases: :class:`MetaSuper`, :class:`MetaSignals`

    Automatic caching of render and rows methods.

    Class variable *no_cache* is a list of names of methods to not cache
    automatically.  Valid method names for *no_cache* are ``'render'`` and
    ``'rows'``.

    Class variable *ignore_focus* if defined and set to ``True`` indicates
    that the canvas this widget renders is not affected by the focus
    parameter, so it may be ignored when caching.
    """

    def __init__(cls, name, bases, d):
        no_cache = d.get('no_cache', [])
        super().__init__(name, bases, d)
        if 'render' in d:
            if 'render' not in no_cache:
                render_fn = cache_widget_render(cls)
            else:
                render_fn = nocache_widget_render(cls)
            cls.render = render_fn
        if 'rows' in d and 'rows' not in no_cache:
            cls.rows = cache_widget_rows(cls)
        if 'no_cache' in d:
            del cls.no_cache
        if 'ignore_focus' in d:
            del cls.ignore_focus