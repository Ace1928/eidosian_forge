from kivy.tests.common import GraphicUnitTest
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.base import EventLoop
from kivy.clock import Clock
from kivy.tests.common import UTMotionEvent
from time import sleep
from itertools import count
class _TestGrid(GridLayout):

    def __init__(self, **kwargs):
        kwargs['cols'] = 1
        kwargs['spacing'] = 10
        kwargs['size_hint'] = (None, None)
        super(_TestGrid, self).__init__(**kwargs)
        self.bind(minimum_height=self.setter('height'))
        self.bind(minimum_width=self.setter('width'))
        for i in range(10):
            self.add_widget(Label(size_hint=(None, None), height=100, width=1000, text=str(i)))