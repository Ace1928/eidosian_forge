import pyglet.gl as pgl
from pyglet import font
from sympy.core import S
from sympy.plotting.pygletplot.plot_object import PlotObject
from sympy.plotting.pygletplot.util import billboard_matrix, dot_product, \
from sympy.utilities.iterables import is_sequence
class PlotAxes(PlotObject):

    def __init__(self, *args, style='', none=None, frame=None, box=None, ordinate=None, stride=0.25, visible='', overlay='', colored='', label_axes='', label_ticks='', tick_length=0.1, font_face='Arial', font_size=28, **kwargs):
        style = style.lower()
        if none is not None:
            style = 'none'
        if frame is not None:
            style = 'frame'
        if box is not None:
            style = 'box'
        if ordinate is not None:
            style = 'ordinate'
        if style in ['', 'ordinate']:
            self._render_object = PlotAxesOrdinate(self)
        elif style in ['frame', 'box']:
            self._render_object = PlotAxesFrame(self)
        elif style in ['none']:
            self._render_object = None
        else:
            raise ValueError('Unrecognized axes style %s.' % style)
        try:
            stride = eval(stride)
        except TypeError:
            pass
        if is_sequence(stride):
            if len(stride) != 3:
                raise ValueError('length should be equal to 3')
            self._stride = stride
        else:
            self._stride = [stride, stride, stride]
        self._tick_length = float(tick_length)
        self._origin = [0, 0, 0]
        self.reset_bounding_box()

        def flexible_boolean(input, default):
            if input in [True, False]:
                return input
            if input in ('f', 'F', 'false', 'False'):
                return False
            if input in ('t', 'T', 'true', 'True'):
                return True
            return default
        self.visible = flexible_boolean(kwargs, True)
        self._overlay = flexible_boolean(overlay, True)
        self._colored = flexible_boolean(colored, False)
        self._label_axes = flexible_boolean(label_axes, False)
        self._label_ticks = flexible_boolean(label_ticks, True)
        self.font_face = font_face
        self.font_size = font_size
        self.reset_resources()

    def reset_resources(self):
        self.label_font = None

    def reset_bounding_box(self):
        self._bounding_box = [[None, None], [None, None], [None, None]]
        self._axis_ticks = [[], [], []]

    def draw(self):
        if self._render_object:
            pgl.glPushAttrib(pgl.GL_ENABLE_BIT | pgl.GL_POLYGON_BIT | pgl.GL_DEPTH_BUFFER_BIT)
            if self._overlay:
                pgl.glDisable(pgl.GL_DEPTH_TEST)
            self._render_object.draw()
            pgl.glPopAttrib()

    def adjust_bounds(self, child_bounds):
        b = self._bounding_box
        c = child_bounds
        for i in range(3):
            if abs(c[i][0]) is S.Infinity or abs(c[i][1]) is S.Infinity:
                continue
            b[i][0] = c[i][0] if b[i][0] is None else min([b[i][0], c[i][0]])
            b[i][1] = c[i][1] if b[i][1] is None else max([b[i][1], c[i][1]])
            self._bounding_box = b
            self._recalculate_axis_ticks(i)

    def _recalculate_axis_ticks(self, axis):
        b = self._bounding_box
        if b[axis][0] is None or b[axis][1] is None:
            self._axis_ticks[axis] = []
        else:
            self._axis_ticks[axis] = strided_range(b[axis][0], b[axis][1], self._stride[axis])

    def toggle_visible(self):
        self.visible = not self.visible

    def toggle_colors(self):
        self._colored = not self._colored