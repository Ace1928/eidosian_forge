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
class BPythonEdit(urwid.Edit):
    """Customized editor *very* tightly interwoven with URWIDRepl.

    Changes include:

    - The edit text supports markup, not just the caption.
      This works by calling set_edit_markup from the change event
      as well as whenever markup changes while text does not.

    - The widget can be made readonly, which currently just means
      it is no longer selectable and stops drawing the cursor.

      This is currently a one-way operation, but that is just because
      I only need and test the readwrite->readonly transition.

    - move_cursor_to_coords is ignored
      (except for internal calls from keypress or mouse_event).

    - arrow up/down are ignored.

    - an "edit-pos-changed" signal is emitted when edit_pos changes.
    """
    signals = ['edit-pos-changed']

    def __init__(self, config, *args, **kwargs):
        self._bpy_text = ''
        self._bpy_attr = []
        self._bpy_selectable = True
        self._bpy_may_move_cursor = False
        self.config = config
        self.tab_length = config.tab_length
        super().__init__(*args, **kwargs)

    def set_edit_pos(self, pos):
        super().set_edit_pos(pos)
        self._emit('edit-pos-changed', self.edit_pos)

    def get_edit_pos(self):
        return self._edit_pos
    edit_pos = property(get_edit_pos, set_edit_pos)

    def make_readonly(self):
        self._bpy_selectable = False
        self._invalidate()

    def set_edit_markup(self, markup):
        """Call this when markup changes but the underlying text does not.

        You should arrange for this to be called from the 'change' signal.
        """
        if markup:
            self._bpy_text, self._bpy_attr = urwid.decompose_tagmarkup(markup)
        else:
            self._bpy_text, self._bpy_attr = ('', [])
        self._invalidate()

    def get_text(self):
        return (self._caption + self._bpy_text, self._attrib + self._bpy_attr)

    def selectable(self):
        return self._bpy_selectable

    def get_cursor_coords(self, *args, **kwargs):
        if not self._bpy_selectable:
            return None
        return super().get_cursor_coords(*args, **kwargs)

    def render(self, size, focus=False):
        if not self._bpy_selectable:
            focus = False
        return super().render(size, focus=focus)

    def get_pref_col(self, size):
        if not self._bpy_selectable:
            return 'left'
        return super().get_pref_col(size)

    def move_cursor_to_coords(self, *args):
        if self._bpy_may_move_cursor:
            return super().move_cursor_to_coords(*args)
        return False

    def keypress(self, size, key):
        if urwid.command_map[key] in ('cursor up', 'cursor down'):
            return key
        self._bpy_may_move_cursor = True
        try:
            if urwid.command_map[key] == 'cursor max left':
                self.edit_pos = 0
            elif urwid.command_map[key] == 'cursor max right':
                self.edit_pos = len(self.get_edit_text())
            elif urwid.command_map[key] == 'clear word':
                if self.edit_pos == 0:
                    return
                line = self.get_edit_text()
                p = len(line[:self.edit_pos].strip())
                line = line[:p] + line[self.edit_pos:]
                np = line.rfind(' ', 0, p)
                if np == -1:
                    line = line[p:]
                    np = 0
                else:
                    line = line[:np] + line[p:]
                self.set_edit_text(line)
                self.edit_pos = np
            elif urwid.command_map[key] == 'clear line':
                line = self.get_edit_text()
                self.set_edit_text(line[self.edit_pos:])
                self.edit_pos = 0
            elif key == 'backspace':
                line = self.get_edit_text()
                cpos = len(line) - self.edit_pos
                if not (cpos or len(line) % self.tab_length or line.strip()):
                    self.set_edit_text(line[:-self.tab_length])
                else:
                    return super().keypress(size, key)
            else:
                return super().keypress(size, key)
            return None
        finally:
            self._bpy_may_move_cursor = False

    def mouse_event(self, *args):
        self._bpy_may_move_cursor = True
        try:
            return super().mouse_event(*args)
        finally:
            self._bpy_may_move_cursor = False