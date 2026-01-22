from json import load
from os.path import exists
from kivy.properties import ObjectProperty, StringProperty, BooleanProperty, \
from kivy.animation import Animation
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.progressbar import ProgressBar
from kivy.uix.label import Label
from kivy.uix.video import Video
from kivy.uix.video import Image
from kivy.factory import Factory
from kivy.logger import Logger
from kivy.clock import Clock
def _load_thumbnail(self, thumbnail):
    if not self.container:
        return
    self.container.clear_widgets()
    if thumbnail:
        self._image = VideoPlayerPreview(source=thumbnail, video=self)
        self.container.add_widget(self._image)