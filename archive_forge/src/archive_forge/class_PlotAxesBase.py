import pyglet.gl as pgl
from pyglet import font
from sympy.core import S
from sympy.plotting.pygletplot.plot_object import PlotObject
from sympy.plotting.pygletplot.util import billboard_matrix, dot_product, \
from sympy.utilities.iterables import is_sequence
class PlotAxesBase(PlotObject):

    def __init__(self, parent_axes):
        self._p = parent_axes

    def draw(self):
        color = [([0.2, 0.1, 0.3], [0.2, 0.1, 0.3], [0.2, 0.1, 0.3]), ([0.9, 0.3, 0.5], [0.5, 1.0, 0.5], [0.3, 0.3, 0.9])][self._p._colored]
        self.draw_background(color)
        self.draw_axis(2, color[2])
        self.draw_axis(1, color[1])
        self.draw_axis(0, color[0])

    def draw_background(self, color):
        pass

    def draw_axis(self, axis, color):
        raise NotImplementedError()

    def draw_text(self, text, position, color, scale=1.0):
        if len(color) == 3:
            color = (color[0], color[1], color[2], 1.0)
        if self._p.label_font is None:
            self._p.label_font = font.load(self._p.font_face, self._p.font_size, bold=True, italic=False)
        label = font.Text(self._p.label_font, text, color=color, valign=font.Text.BASELINE, halign=font.Text.CENTER)
        pgl.glPushMatrix()
        pgl.glTranslatef(*position)
        billboard_matrix()
        scale_factor = 0.005 * scale
        pgl.glScalef(scale_factor, scale_factor, scale_factor)
        pgl.glColor4f(0, 0, 0, 0)
        label.draw()
        pgl.glPopMatrix()

    def draw_line(self, v, color):
        o = self._p._origin
        pgl.glBegin(pgl.GL_LINES)
        pgl.glColor3f(*color)
        pgl.glVertex3f(v[0][0] + o[0], v[0][1] + o[1], v[0][2] + o[2])
        pgl.glVertex3f(v[1][0] + o[0], v[1][1] + o[1], v[1][2] + o[2])
        pgl.glEnd()