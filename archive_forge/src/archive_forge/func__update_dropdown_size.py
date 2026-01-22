from kivy.compat import string_types
from kivy.factory import Factory
from kivy.properties import ListProperty, ObjectProperty, BooleanProperty
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
def _update_dropdown_size(self, *largs):
    if not self.sync_height:
        return
    dp = self._dropdown
    if not dp:
        return
    container = dp.container
    if not container:
        return
    h = self.height
    for item in container.children[:]:
        item.height = h