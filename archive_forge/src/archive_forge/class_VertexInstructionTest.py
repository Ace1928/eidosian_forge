import sys
import pytest
import itertools
from threading import Thread
from kivy.tests.common import GraphicUnitTest, requires_graphics
class VertexInstructionTest(GraphicUnitTest):

    def test_circle(self):
        from kivy.uix.widget import Widget
        from kivy.graphics import Ellipse, Color
        r = self.render
        wid = Widget()
        with wid.canvas:
            Color(1, 1, 1)
            Ellipse(pos=(100, 100), size=(100, 100))
        r(wid)
        wid = Widget()
        with wid.canvas:
            Color(1, 1, 1)
            Ellipse(pos=(100, 100), size=(100, 100), segments=10)
        r(wid)
        wid = Widget()
        with wid.canvas:
            Color(1, 1, 1)
            self.e = Ellipse(pos=(100, 100), size=(100, 100))
        self.e.pos = (10, 10)
        r(wid)

    def test_ellipse(self):
        from kivy.uix.widget import Widget
        from kivy.graphics import Ellipse, Color
        r = self.render
        wid = Widget()
        with wid.canvas:
            Color(1, 1, 1)
            self.e = Ellipse(pos=(100, 100), size=(200, 100))
        r(wid)

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

    def test_line_rounded_rectangle(self):
        from kivy.uix.widget import Widget
        from kivy.graphics import Line, Color
        r = self.render
        wid = Widget()
        with wid.canvas:
            Color(1, 1, 1)
            line = Line(rounded_rectangle=(100, 100, 100, 100, 10, 20, 30, 40, 100))
        r(wid)
        assert line.rounded_rectangle == (100, 100, 100, 100, 10, 20, 30, 40, 100)
        wid = Widget()
        with wid.canvas:
            Color(1, 1, 1)
            line = Line(rounded_rectangle=(100, 100, 100, 100, 100, 20, 10, 30, 100))
        r(wid)
        assert line.rounded_rectangle == (100, 100, 100, 100, 70, 20, 10, 30, 100)
        wid = Widget()
        with wid.canvas:
            Color(1, 1, 1)
            line = Line(rounded_rectangle=(100, 100, 100, 100, 100, 25, 100, 50, 100))
        r(wid)
        assert line.rounded_rectangle == (100, 100, 100, 100, 50, 25, 50, 50, 100)
        wid = Widget()
        with wid.canvas:
            Color(1, 1, 1)
            line = Line(rounded_rectangle=(100, 100, 100, 100, 150, 50, 50.001, 51, 100))
        r(wid)
        assert line.rounded_rectangle == (100, 100, 100, 100, 50, 50, 50, 50, 100)
        wid = Widget()
        with wid.canvas:
            Color(1, 1, 1)
            line = Line(rounded_rectangle=(100, 100, 100, 100, 0, 0, 0, 0, 100))
        r(wid)
        assert line.rounded_rectangle == (100, 100, 100, 100, 1, 1, 1, 1, 100)
        wid = Widget()
        with wid.canvas:
            Color(1, 1, 1)
            line = Line(rounded_rectangle=(100, 100, 100, 100, 100, 0, 0, 0, 100))
        r(wid)
        assert line.rounded_rectangle == (100, 100, 100, 100, 99, 1, 1, 1, 100)

    def test_smoothline_rounded_rectangle(self):
        from kivy.uix.widget import Widget
        from kivy.graphics import SmoothLine, Color
        r = self.render
        wid = Widget()
        with wid.canvas:
            Color(1, 1, 1)
            line = SmoothLine(rounded_rectangle=(100, 100, 0.5, 1.99, 30, 30, 30, 30, 100))
        r(wid)
        assert line.rounded_rectangle is None

    def test_enlarged_line(self):
        from kivy.uix.widget import Widget
        from kivy.graphics import Line, Color, PushMatrix, PopMatrix, Scale, Translate
        r = self.render
        wid = Widget()
        with wid.canvas:
            Color(1, 1, 1)
            Line(points=(10, 10, 10, 90), width=1)
            Line(points=(20, 10, 20, 90), width=3)
            PushMatrix()
            Translate(30, 10, 1)
            Scale(3, 1, 1)
            Line(points=(0, 0, 0, 80), width=1, force_custom_drawing_method=True)
            PopMatrix()
        r(wid)