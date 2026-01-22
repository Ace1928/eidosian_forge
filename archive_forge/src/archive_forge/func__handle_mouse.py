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
def _handle_mouse(self, ev):
    """
        Handle mouse events. Return a list of KeyPress instances.
        """
    FROM_LEFT_1ST_BUTTON_PRESSED = 1
    result = []
    if ev.ButtonState == FROM_LEFT_1ST_BUTTON_PRESSED:
        for event_type in [MouseEventType.MOUSE_DOWN, MouseEventType.MOUSE_UP]:
            data = ';'.join([event_type, str(ev.MousePosition.X), str(ev.MousePosition.Y)])
            result.append(KeyPress(Keys.WindowsMouseEvent, data))
    return result