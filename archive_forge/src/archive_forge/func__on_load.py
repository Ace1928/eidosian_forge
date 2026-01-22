from kivy.clock import Clock
from kivy.uix.image import Image
from kivy.core.video import Video as CoreVideo
from kivy.resources import resource_find
from kivy.properties import (BooleanProperty, NumericProperty, ObjectProperty,
def _on_load(self, *largs):
    self.loaded = True
    self._on_video_frame(largs)