from sympy.utilities.lambdify import lambdify
from sympy.core.numbers import pi
from sympy.functions import sin, cos
from sympy.plotting.pygletplot.plot_curve import PlotCurve
from sympy.plotting.pygletplot.plot_surface import PlotSurface
from math import sin as p_sin
from math import cos as p_cos
class ParametricSurface(PlotSurface):
    i_vars, d_vars = ('uv', 'xyz')
    intervals = [[-1, 1, 40], [-1, 1, 40]]
    aliases = ['parametric']
    is_default = True

    def _get_sympy_evaluator(self):
        fx, fy, fz = self.d_vars
        u = self.u_interval.v
        v = self.v_interval.v

        @float_vec3
        def e(_u, _v):
            return (fx.subs(u, _u).subs(v, _v), fy.subs(u, _u).subs(v, _v), fz.subs(u, _u).subs(v, _v))
        return e

    def _get_lambda_evaluator(self):
        fx, fy, fz = self.d_vars
        u = self.u_interval.v
        v = self.v_interval.v
        return lambdify([u, v], [fx, fy, fz])