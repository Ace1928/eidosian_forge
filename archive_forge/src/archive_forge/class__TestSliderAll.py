from kivy.tests.common import GraphicUnitTest, UTMotionEvent
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.slider import Slider
from kivy.base import EventLoop
class _TestSliderAll(Slider):

    def __init__(self, **kwargs):
        super(_TestSliderAll, self).__init__(**kwargs)
        self.sensitivity = 'all'