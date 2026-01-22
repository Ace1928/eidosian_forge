from kivy.uix.widget import Widget
from kivy.core.image import Image as CoreImage
from kivy.resources import resource_find
from kivy.properties import (
from kivy.logger import Logger
def _update_fit_mode(self, *args):
    keep_ratio = self.keep_ratio
    allow_stretch = self.allow_stretch
    if not keep_ratio and (not allow_stretch) or (keep_ratio and (not allow_stretch)):
        self.fit_mode = 'scale-down'
    elif not keep_ratio and allow_stretch:
        self.fit_mode = 'fill'
    elif keep_ratio and allow_stretch:
        self.fit_mode = 'contain'