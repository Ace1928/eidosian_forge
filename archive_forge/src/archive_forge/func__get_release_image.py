import pyglet
from pyglet.event import EventDispatcher
from pyglet.graphics import Group
from pyglet.text.caret import Caret
from pyglet.text.layout import IncrementalTextLayout
def _get_release_image(self, x, y):
    return self._hover_img if self._check_hit(x, y) else self._depressed_img