from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.properties import BooleanProperty, ListProperty, ObjectProperty, \
def toggle_node(self, node):
    """Toggle the state of the node (open/collapsed).
        """
    node.is_open = not node.is_open
    if node.is_open:
        if self.load_func and (not node.is_loaded):
            self._do_node_load(node)
        self.dispatch('on_node_expand', node)
    else:
        self.dispatch('on_node_collapse', node)
    self._trigger_layout()