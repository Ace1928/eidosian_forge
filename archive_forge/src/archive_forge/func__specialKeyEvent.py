import time
import sys
import AppKit
import pyautogui
from pyautogui import LEFT, MIDDLE, RIGHT
def _specialKeyEvent(key, upDown):
    """ Helper method for special keys.

    Source: http://stackoverflow.com/questions/11045814/emulate-media-key-press-on-mac
    """
    assert upDown in ('up', 'down'), "upDown argument must be 'up' or 'down'"
    key_code = special_key_translate_table[key]
    ev = AppKit.NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(Quartz.NSSystemDefined, (0, 0), 2560 if upDown == 'down' else 2816, 0, 0, 0, 8, key_code << 16 | (10 if upDown == 'down' else 11) << 8, -1)
    Quartz.CGEventPost(0, ev.CGEvent())