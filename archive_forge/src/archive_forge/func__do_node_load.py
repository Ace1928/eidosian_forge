from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.properties import BooleanProperty, ListProperty, ObjectProperty, \
def _do_node_load(self, node):
    gen = self.load_func(self, node)
    if node:
        node.is_loaded = True
    if not gen:
        return
    for cnode in gen:
        self.add_node(cnode, node)