import sys
import os
import time
import locale
import signal
import urwid
from typing import Optional
from . import args as bpargs, repl, translations
from .formatter import theme_map
from .translations import _
from .keys import urwid_key_dispatch as key_dispatch
class Tooltip(urwid.BoxWidget):
    """Container inspired by Overlay to position our tooltip.

    bottom_w should be a BoxWidget.
    The top window currently has to be a listbox to support shrinkwrapping.

    This passes keyboard events to the bottom instead of the top window.

    It also positions the top window relative to the cursor position
    from the bottom window and hides it if there is no cursor.
    """

    def __init__(self, bottom_w, listbox):
        super().__init__()
        self.bottom_w = bottom_w
        self.listbox = listbox
        self.top_w = urwid.LineBox(listbox)
        self.tooltip_focus = False

    def selectable(self):
        return self.bottom_w.selectable()

    def keypress(self, size, key):
        return self.bottom_w.keypress(size, key)

    def mouse_event(self, size, event, button, col, row, focus):
        if not hasattr(self.bottom_w, 'mouse_event'):
            return False
        return self.bottom_w.mouse_event(size, event, button, col, row, focus)

    def get_cursor_coords(self, size):
        return self.bottom_w.get_cursor_coords(size)

    def render(self, size, focus=False):
        maxcol, maxrow = size
        bottom_c = self.bottom_w.render(size, focus)
        cursor = bottom_c.cursor
        if not cursor:
            return bottom_c
        cursor_x, cursor_y = cursor
        if cursor_y * 2 < maxrow:
            y = cursor_y + 1
            rows = maxrow - y
        else:
            y = 0
            rows = cursor_y
        while 'bottom' in self.listbox.ends_visible((maxcol - 2, rows - 3)):
            rows -= 1
        if not y:
            y = cursor_y - rows
        top_c = self.top_w.render((maxcol, rows), focus and self.tooltip_focus)
        combi_c = urwid.CanvasOverlay(top_c, bottom_c, 0, y)
        canvas = urwid.CompositeCanvas(combi_c)
        canvas.cursor = cursor
        return canvas