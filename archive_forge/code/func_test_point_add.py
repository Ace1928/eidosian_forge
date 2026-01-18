import sys
import pytest
import itertools
from threading import Thread
from kivy.tests.common import GraphicUnitTest, requires_graphics
def test_point_add(self):
    from kivy.uix.widget import Widget
    from kivy.graphics import Point, Color
    r = self.render
    wid = Widget()
    with wid.canvas:
        Color(1, 1, 1)
        p = Point(pointsize=10)
    p.add_point(10, 10)
    p.add_point(90, 10)
    p.add_point(10, 90)
    p.add_point(50, 50)
    p.add_point(10, 50)
    p.add_point(50, 10)
    r(wid)