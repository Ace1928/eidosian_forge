from kivy.clock import Clock
from kivy.uix.image import Image
from kivy.core.video import Video as CoreVideo
from kivy.resources import resource_find
from kivy.properties import (BooleanProperty, NumericProperty, ObjectProperty,
def texture_update(self, *largs):
    if self.preview:
        self.set_texture_from_resource(self.preview)
    else:
        self.set_texture_from_resource(self.source)