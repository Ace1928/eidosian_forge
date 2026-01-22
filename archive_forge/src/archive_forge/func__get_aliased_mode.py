from .plot_interval import PlotInterval
from .plot_object import PlotObject
from .util import parse_option_string
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.geometry.entity import GeometryEntity
from sympy.utilities.iterables import is_sequence
@staticmethod
def _get_aliased_mode(alias, i, d, i_vars=-1):
    if i_vars == -1:
        i_vars = i
    if alias not in PlotMode._mode_alias_list:
        raise ValueError("Couldn't find a mode called %s. Known modes: %s." % (alias, ', '.join(PlotMode._mode_alias_list)))
    try:
        return PlotMode._mode_map[d][i][alias]
    except TypeError:
        if i < PlotMode._i_var_max:
            return PlotMode._get_aliased_mode(alias, i + 1, d, i_vars)
        else:
            raise ValueError("Couldn't find a %s mode for %i independent and %i dependent variables." % (alias, i_vars, d))