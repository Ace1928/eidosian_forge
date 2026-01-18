import sys
import pytest
import itertools
from threading import Thread
from kivy.tests.common import GraphicUnitTest, requires_graphics
def test_smoothellipse(self):
    from kivy.uix.widget import Widget
    from kivy.graphics import Color, SmoothEllipse
    r = self.render
    wid = Widget()
    with wid.canvas:
        Color(1, 1, 1, 0.5)
        ellipse = SmoothEllipse(pos=(100, 100), size=(150, 150))
    r(wid)
    ellipse_center = [ellipse.pos[0] + ellipse.size[0] / 2, ellipse.pos[1] + ellipse.size[1] / 2]
    filtered_points = self._filtered_points(ellipse.points + ellipse_center)
    assert ellipse.antialiasing_line_points == filtered_points + filtered_points[:2]
    ellipse.size = (150, -2)
    r(wid)
    assert ellipse.antialiasing_line_points == []
    ellipse.size = (150, 2)
    r(wid)
    assert ellipse.antialiasing_line_points == []
    ellipse.size = (150, 150)
    r(wid)
    assert ellipse.antialiasing_line_points == filtered_points + filtered_points[:2]
    ellipse.texture = self._get_texture()
    r(wid)
    assert ellipse.antialiasing_line_points == []
    ellipse.source = ''
    r(wid)
    assert ellipse.antialiasing_line_points == filtered_points + filtered_points[:2]
    wid = Widget()
    with wid.canvas:
        Color(1, 1, 1, 0.5)
        ellipse = SmoothEllipse(pos=(100, 100), size=(150, 150), angle_start=90, angle_end=-120)
    r(wid)
    ellipse_center = [ellipse.pos[0] + ellipse.size[0] / 2, ellipse.pos[1] + ellipse.size[1] / 2]
    filtered_points = self._filtered_points(ellipse.points + ellipse_center)
    assert ellipse.antialiasing_line_points == filtered_points + filtered_points[:2]
    wid = Widget()
    with wid.canvas:
        Color(1, 1, 1, 0.5)
        ellipse = SmoothEllipse(pos=(100, 100), size=(150, -3))
    r(wid)
    assert ellipse.antialiasing_line_points == []
    wid = Widget()
    with wid.canvas:
        Color(1, 1, 1, 0.5)
        ellipse = SmoothEllipse(pos=(100, 100), size=(3.99, 3.99))
    r(wid)
    assert ellipse.antialiasing_line_points == []