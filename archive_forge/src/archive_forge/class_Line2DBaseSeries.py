from collections.abc import Callable
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import arity, Function
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.external import import_module
from sympy.printing.latex import latex
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from .experimental_lambdify import (vectorized_lambdify, lambdify)
from sympy.plotting.textplot import textplot
class Line2DBaseSeries(BaseSeries):
    """A base class for 2D lines.

    - adding the label, steps and only_integers options
    - making is_2Dline true
    - defining get_segments and get_color_array
    """
    is_2Dline = True
    _dim = 2

    def __init__(self):
        super().__init__()
        self.label = None
        self.steps = False
        self.only_integers = False
        self.line_color = None

    def get_data(self):
        """ Return lists of coordinates for plotting the line.

        Returns
        =======
            x : list
                List of x-coordinates

            y : list
                List of y-coordinates

            z : list
                List of z-coordinates in case of Parametric3DLineSeries
        """
        np = import_module('numpy')
        points = self.get_points()
        if self.steps is True:
            if len(points) == 2:
                x = np.array((points[0], points[0])).T.flatten()[1:]
                y = np.array((points[1], points[1])).T.flatten()[:-1]
                points = (x, y)
            else:
                x = np.repeat(points[0], 3)[2:]
                y = np.repeat(points[1], 3)[:-2]
                z = np.repeat(points[2], 3)[1:-1]
                points = (x, y, z)
        return points

    def get_segments(self):
        sympy_deprecation_warning('\n            The Line2DBaseSeries.get_segments() method is deprecated.\n\n            Instead, use the MatplotlibBackend.get_segments() method, or use\n            The get_points() or get_data() methods.\n            ', deprecated_since_version='1.9', active_deprecations_target='deprecated-get-segments')
        np = import_module('numpy')
        points = type(self).get_data(self)
        points = np.ma.array(points).T.reshape(-1, 1, self._dim)
        return np.ma.concatenate([points[:-1], points[1:]], axis=1)

    def get_color_array(self):
        np = import_module('numpy')
        c = self.line_color
        if hasattr(c, '__call__'):
            f = np.vectorize(c)
            nargs = arity(c)
            if nargs == 1 and self.is_parametric:
                x = self.get_parameter_points()
                return f(centers_of_segments(x))
            else:
                variables = list(map(centers_of_segments, self.get_points()))
                if nargs == 1:
                    return f(variables[0])
                elif nargs == 2:
                    return f(*variables[:2])
                else:
                    return f(*variables)
        else:
            return c * np.ones(self.nb_of_points)