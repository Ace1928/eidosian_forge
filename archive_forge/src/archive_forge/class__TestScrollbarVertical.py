from kivy.tests.common import GraphicUnitTest
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.base import EventLoop
from kivy.clock import Clock
from kivy.tests.common import UTMotionEvent
from time import sleep
from itertools import count
class _TestScrollbarVertical(ScrollView):

    def __init__(self, **kwargs):
        kwargs['scroll_type'] = ['bars']
        kwargs['bar_width'] = 20
        kwargs['do_scroll_x'] = False
        super(_TestScrollbarVertical, self).__init__(**kwargs)