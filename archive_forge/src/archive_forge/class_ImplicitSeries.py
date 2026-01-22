from .plot import BaseSeries, Plot
from .experimental_lambdify import experimental_lambdify, vectorized_lambdify
from .intervalmath import interval
from sympy.core.relational import (Equality, GreaterThan, LessThan,
from sympy.core.containers import Tuple
from sympy.core.relational import Eq
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.external import import_module
from sympy.logic.boolalg import BooleanFunction
from sympy.polys.polyutils import _sort_gens
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import flatten
import warnings
class ImplicitSeries(BaseSeries):
    """ Representation for Implicit plot """
    is_implicit = True

    def __init__(self, expr, var_start_end_x, var_start_end_y, has_equality, use_interval_math, depth, nb_of_points, line_color):
        super().__init__()
        self.expr = sympify(expr)
        self.label = self.expr
        self.var_x = sympify(var_start_end_x[0])
        self.start_x = float(var_start_end_x[1])
        self.end_x = float(var_start_end_x[2])
        self.var_y = sympify(var_start_end_y[0])
        self.start_y = float(var_start_end_y[1])
        self.end_y = float(var_start_end_y[2])
        self.get_points = self.get_raster
        self.has_equality = has_equality
        self.nb_of_points = nb_of_points
        self.use_interval_math = use_interval_math
        self.depth = 4 + depth
        self.line_color = line_color

    def __str__(self):
        return 'Implicit equation: %s for %s over %s and %s over %s' % (str(self.expr), str(self.var_x), str((self.start_x, self.end_x)), str(self.var_y), str((self.start_y, self.end_y)))

    def get_raster(self):
        func = experimental_lambdify((self.var_x, self.var_y), self.expr, use_interval=True)
        xinterval = interval(self.start_x, self.end_x)
        yinterval = interval(self.start_y, self.end_y)
        try:
            func(xinterval, yinterval)
        except AttributeError:
            if self.use_interval_math:
                warnings.warn('Adaptive meshing could not be applied to the expression. Using uniform meshing.', stacklevel=7)
            self.use_interval_math = False
        if self.use_interval_math:
            return self._get_raster_interval(func)
        else:
            return self._get_meshes_grid()

    def _get_raster_interval(self, func):
        """ Uses interval math to adaptively mesh and obtain the plot"""
        k = self.depth
        interval_list = []
        np = import_module('numpy')
        xsample = np.linspace(self.start_x, self.end_x, 33)
        ysample = np.linspace(self.start_y, self.end_y, 33)
        jitterx = (np.random.rand(len(xsample)) * 2 - 1) * (self.end_x - self.start_x) / 2 ** 20
        jittery = (np.random.rand(len(ysample)) * 2 - 1) * (self.end_y - self.start_y) / 2 ** 20
        xsample += jitterx
        ysample += jittery
        xinter = [interval(x1, x2) for x1, x2 in zip(xsample[:-1], xsample[1:])]
        yinter = [interval(y1, y2) for y1, y2 in zip(ysample[:-1], ysample[1:])]
        interval_list = [[x, y] for x in xinter for y in yinter]
        plot_list = []

        def refine_pixels(interval_list):
            """ Evaluates the intervals and subdivides the interval if the
            expression is partially satisfied."""
            temp_interval_list = []
            plot_list = []
            for intervals in interval_list:
                intervalx = intervals[0]
                intervaly = intervals[1]
                func_eval = func(intervalx, intervaly)
                if func_eval[1] is False or func_eval[0] is False:
                    pass
                elif func_eval == (True, True):
                    plot_list.append([intervalx, intervaly])
                elif func_eval[1] is None or func_eval[0] is None:
                    avgx = intervalx.mid
                    avgy = intervaly.mid
                    a = interval(intervalx.start, avgx)
                    b = interval(avgx, intervalx.end)
                    c = interval(intervaly.start, avgy)
                    d = interval(avgy, intervaly.end)
                    temp_interval_list.append([a, c])
                    temp_interval_list.append([a, d])
                    temp_interval_list.append([b, c])
                    temp_interval_list.append([b, d])
            return (temp_interval_list, plot_list)
        while k >= 0 and len(interval_list):
            interval_list, plot_list_temp = refine_pixels(interval_list)
            plot_list.extend(plot_list_temp)
            k = k - 1
        if self.has_equality:
            for intervals in interval_list:
                intervalx = intervals[0]
                intervaly = intervals[1]
                func_eval = func(intervalx, intervaly)
                if func_eval[1] and func_eval[0] is not False:
                    plot_list.append([intervalx, intervaly])
        return (plot_list, 'fill')

    def _get_meshes_grid(self):
        """Generates the mesh for generating a contour.

        In the case of equality, ``contour`` function of matplotlib can
        be used. In other cases, matplotlib's ``contourf`` is used.
        """
        equal = False
        if isinstance(self.expr, Equality):
            expr = self.expr.lhs - self.expr.rhs
            equal = True
        elif isinstance(self.expr, (GreaterThan, StrictGreaterThan)):
            expr = self.expr.lhs - self.expr.rhs
        elif isinstance(self.expr, (LessThan, StrictLessThan)):
            expr = self.expr.rhs - self.expr.lhs
        else:
            raise NotImplementedError('The expression is not supported for plotting in uniform meshed plot.')
        np = import_module('numpy')
        xarray = np.linspace(self.start_x, self.end_x, self.nb_of_points)
        yarray = np.linspace(self.start_y, self.end_y, self.nb_of_points)
        x_grid, y_grid = np.meshgrid(xarray, yarray)
        func = vectorized_lambdify((self.var_x, self.var_y), expr)
        z_grid = func(x_grid, y_grid)
        z_grid[np.ma.where(z_grid < 0)] = -1
        z_grid[np.ma.where(z_grid > 0)] = 1
        if equal:
            return (xarray, yarray, z_grid, 'contour')
        else:
            return (xarray, yarray, z_grid, 'contourf')