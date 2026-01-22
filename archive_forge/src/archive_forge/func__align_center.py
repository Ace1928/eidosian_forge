from kivy.animation import Animation
from kivy.properties import (
from kivy.uix.anchorlayout import AnchorLayout
def _align_center(self, *_args):
    if self._is_open:
        self.center = self._window.center