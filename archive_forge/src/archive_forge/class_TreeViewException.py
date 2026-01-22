from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.properties import BooleanProperty, ListProperty, ObjectProperty, \
class TreeViewException(Exception):
    """Exception for errors in the :class:`TreeView`.
    """
    pass