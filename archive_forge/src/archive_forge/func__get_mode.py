from .plot_interval import PlotInterval
from .plot_object import PlotObject
from .util import parse_option_string
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.geometry.entity import GeometryEntity
from sympy.utilities.iterables import is_sequence
@staticmethod
def _get_mode(mode_arg, i_var_count, d_var_count):
    """
        Tries to return an appropriate mode class.
        Intended to be called only by __new__.

        mode_arg
            Can be a string or a class. If it is a
            PlotMode subclass, it is simply returned.
            If it is a string, it can an alias for
            a mode or an empty string. In the latter
            case, we try to find a default mode for
            the i_var_count and d_var_count.

        i_var_count
            The number of independent variables
            needed to evaluate the d_vars.

        d_var_count
            The number of dependent variables;
            usually the number of functions to
            be evaluated in plotting.

        For example, a Cartesian function y = f(x) has
        one i_var (x) and one d_var (y). A parametric
        form x,y,z = f(u,v), f(u,v), f(u,v) has two
        two i_vars (u,v) and three d_vars (x,y,z).
        """
    try:
        m = None
        if issubclass(mode_arg, PlotMode):
            m = mode_arg
    except TypeError:
        pass
    if m:
        if not m._was_initialized:
            raise ValueError('To use unregistered plot mode %s you must first call %s._init_mode().' % (m.__name__, m.__name__))
        if d_var_count != m.d_var_count:
            raise ValueError('%s can only plot functions with %i dependent variables.' % (m.__name__, m.d_var_count))
        if i_var_count > m.i_var_count:
            raise ValueError('%s cannot plot functions with more than %i independent variables.' % (m.__name__, m.i_var_count))
        return m
    if isinstance(mode_arg, str):
        i, d = (i_var_count, d_var_count)
        if i > PlotMode._i_var_max:
            raise ValueError(var_count_error(True, True))
        if d > PlotMode._d_var_max:
            raise ValueError(var_count_error(False, True))
        if not mode_arg:
            return PlotMode._get_default_mode(i, d)
        else:
            return PlotMode._get_aliased_mode(mode_arg, i, d)
    else:
        raise ValueError('PlotMode argument must be a class or a string')