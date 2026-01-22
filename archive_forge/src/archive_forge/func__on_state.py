from kivy.properties import AliasProperty, StringProperty, ColorProperty
from kivy.uix.behaviors import ToggleButtonBehavior
from kivy.uix.widget import Widget
def _on_state(self, instance, value):
    if self.group and self.state == 'down':
        self._release_group(self)