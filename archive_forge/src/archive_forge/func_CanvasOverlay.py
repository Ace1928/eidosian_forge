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
def CanvasOverlay(top_c: Canvas, bottom_c: Canvas, left: int, top: int) -> CompositeCanvas:
    """
    Overlay canvas top_c onto bottom_c at position (left, top).
    """
    overlayed_canvas = CompositeCanvas(bottom_c)
    overlayed_canvas.overlay(top_c, left, top)
    overlayed_canvas.children = [(left, top, top_c, None), (0, 0, bottom_c, None)]
    overlayed_canvas.shortcuts = {}
    for shortcut in top_c.shortcuts:
        overlayed_canvas.shortcuts[shortcut] = 'fg'
    return overlayed_canvas