from midi devices.  It can also list midi devices on the system.
import math
import atexit
import pygame
import pygame.locals
import pygame.pypm as _pypm
def _check_open(self):
    if self._output is None:
        raise MidiException('midi not open.')
    if self._aborted:
        raise MidiException('midi aborted.')