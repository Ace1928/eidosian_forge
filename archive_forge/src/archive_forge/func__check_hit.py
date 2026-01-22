import pyglet
from pyglet.event import EventDispatcher
from pyglet.graphics import Group
from pyglet.text.caret import Caret
from pyglet.text.layout import IncrementalTextLayout
def _check_hit(self, x, y):
    return self._x < x < self._x + self._width and self._y < y < self._y + self._height