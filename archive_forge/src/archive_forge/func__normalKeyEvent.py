import time
import sys
import AppKit
import pyautogui
from pyautogui import LEFT, MIDDLE, RIGHT
def _normalKeyEvent(key, upDown):
    assert upDown in ('up', 'down'), "upDown argument must be 'up' or 'down'"
    try:
        if pyautogui.isShiftCharacter(key):
            key_code = keyboardMapping[key.lower()]
            event = Quartz.CGEventCreateKeyboardEvent(None, keyboardMapping['shift'], upDown == 'down')
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
            time.sleep(pyautogui.DARWIN_CATCH_UP_TIME)
        else:
            key_code = keyboardMapping[key]
        event = Quartz.CGEventCreateKeyboardEvent(None, key_code, upDown == 'down')
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        time.sleep(pyautogui.DARWIN_CATCH_UP_TIME)
    except KeyError:
        raise RuntimeError('Key %s not implemented.' % key)