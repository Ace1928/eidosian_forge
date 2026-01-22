import pyglet
from pyglet.event import EventDispatcher
from pyglet.graphics import Group
from pyglet.text.caret import Caret
from pyglet.text.layout import IncrementalTextLayout
@property
def _max_y(self):
    return self._y + self._half_knob_height + self._base_img.height / 2