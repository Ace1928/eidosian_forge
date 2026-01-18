import pyglet
from pyglet.event import EventDispatcher
from pyglet.graphics import Group
from pyglet.text.caret import Caret
from pyglet.text.layout import IncrementalTextLayout
def update_groups(self, order):
    self._outline.group = Group(order=order + 1, parent=self._user_group)
    self._layout.group = Group(order=order + 2, parent=self._user_group)