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
class VideoPlayerAnnotation(Label):
    """Annotation class used for creating annotation labels.

    Additional keys are available:

    * bgcolor: [r, g, b, a] - background color of the text box
    * bgsource: 'filename' - background image used for the background text box
    * border: (n, e, s, w) - border used for the background image

    """
    start = NumericProperty(0)
    'Start time of the annotation.\n\n    :attr:`start` is a :class:`~kivy.properties.NumericProperty` and defaults\n    to 0.\n    '
    duration = NumericProperty(1)
    'Duration of the annotation.\n\n    :attr:`duration` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 1.\n    '
    annotation = DictProperty({})

    def on_annotation(self, instance, ann):
        for key, value in ann.items():
            setattr(self, key, value)