from sympy.utilities.lambdify import lambdify
from sympy.core.numbers import pi
from sympy.functions import sin, cos
from sympy.plotting.pygletplot.plot_curve import PlotCurve
from sympy.plotting.pygletplot.plot_surface import PlotSurface
from math import sin as p_sin
from math import cos as p_cos
class Cylindrical(PlotSurface):
    i_vars, d_vars = ('th', 'r')
    intervals = [[0, 2 * pi, 40], [-1, 1, 20]]
    aliases = ['cylindrical', 'polar']
    is_default = False

    def _get_sympy_evaluator(self):
        fr = self.d_vars[0]
        t = self.u_interval.v
        h = self.v_interval.v

        def e(_t, _h):
            _r = float(fr.subs(t, _t).subs(h, _h))
            return (_r * p_cos(_t), _r * p_sin(_t), _h)
        return e

    def _get_lambda_evaluator(self):
        fr = self.d_vars[0]
        t = self.u_interval.v
        h = self.v_interval.v
        fx, fy = (fr * cos(t), fr * sin(t))
        return lambdify([t, h], [fx, fy, h])