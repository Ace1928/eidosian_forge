import ctypes
import ctypes.wintypes
import pyautogui
from pyautogui import LEFT, MIDDLE, RIGHT
import sys
def _keyUp(key):
    """Performs a keyboard key release (without the press down beforehand).

    Args:
      key (str): The key to be released up. The valid names are listed in
      pyautogui.KEY_NAMES.

    Returns:
      None
    """
    if key not in keyboardMapping or keyboardMapping[key] is None:
        return
    needsShift = pyautogui.isShiftCharacter(key)
    '\n    # OLD CODE: The new code relies on having all keys be loaded in keyboardMapping from the start.\n    if key in keyboardMapping.keys():\n        vkCode = keyboardMapping[key]\n    elif len(key) == 1:\n        # note: I could use this case to update keyboardMapping to cache the VkKeyScan results, but I\'ve decided not to just to make any possible bugs easier to reproduce.\n        vkCode = ctypes.windll.user32.VkKeyScanW(ctypes.wintypes.WCHAR(key))\n        if vkCode == -1:\n            raise ValueError(\'There is no VK code for key "%s"\' % (key))\n        if vkCode > 0x100: # the vk code will be > 0x100 if it needs shift\n            vkCode -= 0x100\n            needsShift = True\n    '
    mods, vkCode = divmod(keyboardMapping[key], 256)
    for apply_mod, vk_mod in [(mods & 4, 18), (mods & 2, 17), (mods & 1 or needsShift, 16)]:
        if apply_mod:
            ctypes.windll.user32.keybd_event(vk_mod, 0, 0, 0)
    ctypes.windll.user32.keybd_event(vkCode, 0, KEYEVENTF_KEYUP, 0)
    for apply_mod, vk_mod in [(mods & 1 or needsShift, 16), (mods & 2, 17), (mods & 4, 18)]:
        if apply_mod:
            ctypes.windll.user32.keybd_event(vk_mod, 0, KEYEVENTF_KEYUP, 0)