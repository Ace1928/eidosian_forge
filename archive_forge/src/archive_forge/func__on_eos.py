from kivy.clock import Clock
from kivy.uix.image import Image
from kivy.core.video import Video as CoreVideo
from kivy.resources import resource_find
from kivy.properties import (BooleanProperty, NumericProperty, ObjectProperty,
def _on_eos(self, *largs):
    if not self._video or self._video.eos != 'loop':
        self.state = 'stop'
        self.eos = True