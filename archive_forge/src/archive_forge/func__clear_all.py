from kivy.uix.boxlayout import BoxLayout
from kivy.uix.dropdown import DropDown
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.checkbox import CheckBox
from kivy.uix.spinner import Spinner
from kivy.uix.label import Label
from kivy.config import Config
from kivy.properties import ObjectProperty, NumericProperty, BooleanProperty, \
from kivy.metrics import sp
from kivy.lang import Builder
from functools import partial
def _clear_all(self):
    lst = self._list_action_items[:]
    self.clear_widgets()
    for group in self._list_action_group:
        group.clear_widgets()
    self.overflow_group.clear_widgets()
    self.overflow_group.list_action_item = []
    self._list_action_items = lst