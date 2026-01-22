from kivy.uix.widget import Widget
from kivy.core.image import Image as CoreImage
from kivy.resources import resource_find
from kivy.properties import (
from kivy.logger import Logger
def _on_source_error(self, instance, error=None):
    self.dispatch('on_error', error)