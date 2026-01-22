from kivy.uix.widget import Widget
from kivy.core.image import Image as CoreImage
from kivy.resources import resource_find
from kivy.properties import (
from kivy.logger import Logger
def _clear_core_image(self):
    if self._coreimage:
        self._coreimage.unbind(on_load=self._on_source_load)
    super()._clear_core_image()
    self._found_source = None