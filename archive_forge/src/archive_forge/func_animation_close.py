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
def animation_close(self, instance, value):
    if not self.activated:
        self.inspect_enabled = False
        self.win.remove_widget(self)
        self.content.clear_widgets()
        treeview = self.treeview
        for node in list(treeview.iterate_all_nodes()):
            node.widget_ref = None
            treeview.remove_node(node)
        self._window_node = None
        if self._update_widget_tree_ev is not None:
            self._update_widget_tree_ev.cancel()
        widgettree = self.widgettree
        for node in list(widgettree.iterate_all_nodes()):
            widgettree.remove_node(node)
        Logger.info('Inspector: inspector deactivated')