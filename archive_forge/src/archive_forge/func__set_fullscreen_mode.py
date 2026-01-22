import sys
from typing import Tuple
import pyglet
import pyglet.window.key
import pyglet.window.mouse
from pyglet import gl
from pyglet.math import Mat4
from pyglet.event import EventDispatcher
from pyglet.window import key, event
from pyglet.graphics import shader
def _set_fullscreen_mode(self, mode, width, height):
    if mode is not None:
        self.screen.set_mode(mode)
        if width is None:
            width = self.screen.width
        if height is None:
            height = self.screen.height
    elif width is not None or height is not None:
        if width is None:
            width = 0
        if height is None:
            height = 0
        mode = self.screen.get_closest_mode(width, height)
        if mode is not None:
            self.screen.set_mode(mode)
        elif self.screen.get_modes():
            raise NoSuchScreenModeException(f'No mode matching {width}x{height}')
    else:
        width = self.screen.width
        height = self.screen.height
    return (width, height)