from kivy.event import EventDispatcher
from kivy.eventmanager import (
from kivy.factory import Factory
from kivy.properties import (
from kivy.graphics import (
from kivy.graphics.transformation import Matrix
from kivy.base import EventLoop
from kivy.lang import Builder
from kivy.context import get_current_context
from kivy.weakproxy import WeakProxy
from functools import partial
from itertools import islice
def _walk_reverse(self, loopback=False, go_up=False):
    root = self
    index = 0
    if go_up:
        root = self.parent
        try:
            if root is None or not isinstance(root, Widget):
                raise ValueError
            index = root.children.index(self) + 1
        except ValueError:
            if not loopback:
                return
            index = 0
            go_up = False
            root = self
    for child in islice(root.children, index, None):
        for walk_child in child._walk_reverse(loopback=loopback):
            yield walk_child
    yield root
    if go_up:
        for walk_child in root._walk_reverse(loopback=loopback, go_up=go_up):
            yield walk_child