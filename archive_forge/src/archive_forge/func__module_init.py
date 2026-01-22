from midi devices.  It can also list midi devices on the system.
import math
import atexit
import pygame
import pygame.locals
import pygame.pypm as _pypm
def _module_init(state=None):
    if state is not None:
        _module_init.value = state
        return state
    try:
        _module_init.value
    except AttributeError:
        return False
    return _module_init.value