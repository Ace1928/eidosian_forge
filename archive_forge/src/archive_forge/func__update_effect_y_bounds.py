from functools import partial
from kivy.animation import Animation
from kivy.compat import string_types
from kivy.config import Config
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.uix.stencilview import StencilView
from kivy.metrics import dp
from kivy.effects.dampedscroll import DampedScrollEffect
from kivy.properties import NumericProperty, BooleanProperty, AliasProperty, \
from kivy.uix.behaviors import FocusBehavior
def _update_effect_y_bounds(self, *args):
    if not self._viewport or not self.effect_y:
        return
    scrollable_height = self.height - self.viewport_size[1]
    self.effect_y.min = 0 if scrollable_height < 0 else scrollable_height
    self.effect_y.max = scrollable_height
    self.effect_y.value = self.effect_y.max * self.scroll_y