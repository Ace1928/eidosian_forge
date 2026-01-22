import math
from abc import ABC, abstractmethod
import pyglet
from pyglet.gl import GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_BLEND, GL_TRIANGLES
from pyglet.gl import glBlendFunc, glEnable, glDisable
from pyglet.graphics import Batch, Group
from pyglet.math import Vec2
class BezierCurve(ShapeBase):

    def __init__(self, *points, t=1.0, segments=100, thickness=1, color=(255, 255, 255, 255), batch=None, group=None):
        """Create a Bézier curve.

        The curve's anchor point (x, y) defaults to its first control point.

        :Parameters:
            `points` : List[[int, int]]
                Control points of the curve. Points can be specified as multiple
                lists or tuples of point pairs. Ex. (0,0), (2,3), (1,9)
            `t` : float
                Draw `100*t` percent of the curve. 0.5 means the curve
                is half drawn and 1.0 means draw the whole curve.
            `segments` : int
                You can optionally specify how many line segments the
                curve should be made from.
            `thickness` : float
                The desired thickness or width of the line used for the curve.
            `color` : (int, int, int, int)
                The RGB or RGBA color of the curve, specified as a
                tuple of 3 or 4 ints in the range of 0-255. RGB colors
                will be treated as having an opacity of 255.
            `batch` : `~pyglet.graphics.Batch`
                Optional batch to add the curve to.
            `group` : `~pyglet.graphics.Group`
                Optional parent group of the curve.
        """
        self._points = list(points)
        self._x, self._y = self._points[0]
        self._t = t
        self._segments = segments
        self._thickness = thickness
        self._num_verts = self._segments * 6
        r, g, b, *a = color
        self._rgba = (r, g, b, a[0] if a else 255)
        program = get_default_shader()
        self._batch = batch or Batch()
        self._group = self.group_class(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, program, group)
        self._create_vertex_list()

    def _make_curve(self, t):
        n = len(self._points) - 1
        p = [0, 0]
        for i in range(n + 1):
            m = math.comb(n, i) * (1 - t) ** (n - i) * t ** i
            p[0] += m * self._points[i][0]
            p[1] += m * self._points[i][1]
        return p

    def _create_vertex_list(self):
        self._vertex_list = self._group.program.vertex_list(self._num_verts, self._draw_mode, self._batch, self._group, position=('f', self._get_vertices()), colors=('Bn', self._rgba * self._num_verts), translation=('f', (self._x, self._y) * self._num_verts))

    def _get_vertices(self):
        if not self._visible:
            return (0, 0) * self._num_verts
        else:
            x = -self._anchor_x - self._x
            y = -self._anchor_y - self._y
            points = [(x + self._make_curve(self._t * t / self._segments)[0], y + self._make_curve(self._t * t / self._segments)[1]) for t in range(self._segments + 1)]
            trans_x, trans_y = points[0]
            trans_x += self._anchor_x
            trans_y += self._anchor_y
            coords = [[x - trans_x, y - trans_y] for x, y in points]
            vertices = []
            prev_miter = None
            prev_scale = None
            for i in range(len(coords) - 1):
                prev_point = None
                next_point = None
                if i > 0:
                    prev_point = points[i - 1]
                if i + 2 < len(points):
                    next_point = points[i + 2]
                prev_miter, prev_scale, *segment = _get_segment(prev_point, points[i], points[i + 1], next_point, self._thickness, prev_miter, prev_scale)
                vertices.extend(segment)
            return vertices

    def _update_vertices(self):
        self._vertex_list.position[:] = self._get_vertices()

    @property
    def points(self):
        """Control points of the curve.

        :type: List[[int, int]]
        """
        return self._points

    @points.setter
    def points(self, value):
        self._points = value
        self._update_vertices()

    @property
    def t(self):
        """Draw `100*t` percent of the curve.

        :type: float
        """
        return self._t

    @t.setter
    def t(self, value):
        self._t = value
        self._update_vertices()

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, thickness):
        self._thickness = thickness
        self._update_vertices()