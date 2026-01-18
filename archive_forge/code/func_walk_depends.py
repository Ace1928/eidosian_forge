from __future__ import annotations
import contextlib
import dataclasses
import typing
import warnings
import weakref
from contextlib import suppress
from urwid.str_util import calc_text_pos, calc_width
from urwid.text_layout import LayoutSegment, trim_line
from urwid.util import (
def walk_depends(canv):
    """
            Collect all child widgets for determining who we
            depend on.
            """
    depends = []
    for _x, _y, c, _pos in canv.children:
        if c.widget_info:
            depends.append(c.widget_info[0])
        elif hasattr(c, 'children'):
            depends.extend(walk_depends(c))
    return depends