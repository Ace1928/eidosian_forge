import weakref
from functools import partial
from itertools import chain
from kivy.animation import Animation
from kivy.logger import Logger
from kivy.graphics.transformation import Matrix
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.factory import Factory
from kivy.weakproxy import WeakProxy
from kivy.properties import (
def update_node_content(self, node, *largs):
    node = node()
    if node is None:
        return
    node.refresh = True
    node.refresh = False