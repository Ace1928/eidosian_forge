import sys
import pytest
import itertools
from threading import Thread
from kivy.tests.common import GraphicUnitTest, requires_graphics
def test_smoothquad(self):
    from kivy.uix.widget import Widget
    from kivy.graphics import Color, SmoothQuad
    r = self.render
    wid = Widget()
    with wid.canvas:
        Color(1, 0, 0, 0.5)
        quad = SmoothQuad(points=[100, 100, 100, 200, 200, 200, 200, 100])
    r(wid)
    filtered_points = self._filtered_points(quad.points)
    assert quad.antialiasing_line_points == filtered_points + filtered_points[:2]
    wid = Widget()
    with wid.canvas:
        Color(1, 0, 0, 0.5)
        quad = SmoothQuad(points=[200, 100, 200, 200, 100, 200, 100, 100])
    r(wid)
    filtered_points = self._filtered_points(quad.points)
    assert quad.antialiasing_line_points == filtered_points + filtered_points[:2]
    wid = Widget()
    with wid.canvas:
        Color(0, 1, 0, 0.5)
        quad = SmoothQuad(points=[200, 100, 200, 100.8, 100, 100.8, 100, 100])
    r(wid)
    assert quad.antialiasing_line_points == []
    quad.points = [200, 100, 200, 200, 100, 200, 100, 100]
    r(wid)
    assert quad.antialiasing_line_points == filtered_points + filtered_points[:2]
    quad.texture = self._get_texture()
    r(wid)
    assert quad.antialiasing_line_points == []
    quad.source = ''
    r(wid)
    assert quad.antialiasing_line_points == filtered_points + filtered_points[:2]