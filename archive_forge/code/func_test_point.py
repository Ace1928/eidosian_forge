import sys
import pytest
import itertools
from threading import Thread
from kivy.tests.common import GraphicUnitTest, requires_graphics
def test_point(self):
    from kivy.uix.widget import Widget
    from kivy.graphics import Point, Color
    r = self.render
    wid = Widget()
    with wid.canvas:
        Color(1, 1, 1)
        Point(points=(10, 10))
    r(wid)
    wid = Widget()
    with wid.canvas:
        Color(1, 1, 1)
        Point(points=[x * 5 for x in range(50)])
    r(wid)