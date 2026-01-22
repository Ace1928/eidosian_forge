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
class SurfaceOver2DRangeSeries(SurfaceBaseSeries):
    """Representation for a 3D surface consisting of a SymPy expression and 2D
    range."""

    def __init__(self, expr, var_start_end_x, var_start_end_y, **kwargs):
        super().__init__()
        self.expr = sympify(expr)
        self.var_x = sympify(var_start_end_x[0])
        self.start_x = float(var_start_end_x[1])
        self.end_x = float(var_start_end_x[2])
        self.var_y = sympify(var_start_end_y[0])
        self.start_y = float(var_start_end_y[1])
        self.end_y = float(var_start_end_y[2])
        self.nb_of_points_x = kwargs.get('nb_of_points_x', 50)
        self.nb_of_points_y = kwargs.get('nb_of_points_y', 50)
        self.surface_color = kwargs.get('surface_color', None)
        self._xlim = (self.start_x, self.end_x)
        self._ylim = (self.start_y, self.end_y)

    def __str__(self):
        return 'cartesian surface: %s for %s over %s and %s over %s' % (str(self.expr), str(self.var_x), str((self.start_x, self.end_x)), str(self.var_y), str((self.start_y, self.end_y)))

    def get_meshes(self):
        np = import_module('numpy')
        mesh_x, mesh_y = np.meshgrid(np.linspace(self.start_x, self.end_x, num=self.nb_of_points_x), np.linspace(self.start_y, self.end_y, num=self.nb_of_points_y))
        f = vectorized_lambdify((self.var_x, self.var_y), self.expr)
        mesh_z = f(mesh_x, mesh_y)
        mesh_z = np.array(mesh_z, dtype=np.float64)
        mesh_z = np.ma.masked_invalid(mesh_z)
        self._zlim = (np.amin(mesh_z), np.amax(mesh_z))
        return (mesh_x, mesh_y, mesh_z)