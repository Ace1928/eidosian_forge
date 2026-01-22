from __future__ import unicode_literals
from ctypes import windll, pointer
from ctypes.wintypes import DWORD
from six.moves import range
from prompt_toolkit.key_binding.input_processor import KeyPress
from prompt_toolkit.keys import Keys
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.win32_types import EventTypes, KEY_EVENT_RECORD, MOUSE_EVENT_RECORD, INPUT_RECORD, STD_INPUT_HANDLE
import msvcrt
import os
import sys
import six
def _event_to_key_presses(self, ev):
    """
        For this `KEY_EVENT_RECORD`, return a list of `KeyPress` instances.
        """
    assert type(ev) == KEY_EVENT_RECORD and ev.KeyDown
    result = None
    u_char = ev.uChar.UnicodeChar
    ascii_char = u_char.encode('utf-8')
    if u_char == '\x00':
        if ev.VirtualKeyCode in self.keycodes:
            result = KeyPress(self.keycodes[ev.VirtualKeyCode], '')
    elif ascii_char in self.mappings:
        if self.mappings[ascii_char] == Keys.ControlJ:
            u_char = '\n'
        result = KeyPress(self.mappings[ascii_char], u_char)
    else:
        result = KeyPress(u_char, u_char)
    if (ev.ControlKeyState & self.LEFT_CTRL_PRESSED or ev.ControlKeyState & self.RIGHT_CTRL_PRESSED) and result:
        if result.key == Keys.Left:
            result.key = Keys.ControlLeft
        if result.key == Keys.Right:
            result.key = Keys.ControlRight
        if result.key == Keys.Up:
            result.key = Keys.ControlUp
        if result.key == Keys.Down:
            result.key = Keys.ControlDown
    if ev.ControlKeyState & self.SHIFT_PRESSED and result:
        if result.key == Keys.Tab:
            result.key = Keys.BackTab
    if (ev.ControlKeyState & self.LEFT_CTRL_PRESSED or ev.ControlKeyState & self.RIGHT_CTRL_PRESSED) and result and (result.data == ' '):
        result = KeyPress(Keys.ControlSpace, ' ')
    if (ev.ControlKeyState & self.LEFT_CTRL_PRESSED or ev.ControlKeyState & self.RIGHT_CTRL_PRESSED) and result and (result.key == Keys.ControlJ):
        return [KeyPress(Keys.Escape, ''), result]
    if result:
        meta_pressed = ev.ControlKeyState & self.LEFT_ALT_PRESSED
        if meta_pressed:
            return [KeyPress(Keys.Escape, ''), result]
        else:
            return [result]
    else:
        return []