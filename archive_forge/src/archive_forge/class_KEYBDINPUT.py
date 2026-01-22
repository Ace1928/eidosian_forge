import ctypes
import ctypes.wintypes
import pyautogui
from pyautogui import LEFT, MIDDLE, RIGHT
import sys
class KEYBDINPUT(ctypes.Structure):
    _fields_ = [('wVk', ctypes.wintypes.WORD), ('wScan', ctypes.wintypes.WORD), ('dwFlags', ctypes.wintypes.DWORD), ('time', ctypes.wintypes.DWORD), ('dwExtraInfo', ctypes.POINTER(ctypes.wintypes.ULONG))]