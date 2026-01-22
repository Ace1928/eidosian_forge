import itertools
from pyomo.common.dependencies import (
from pyomo.common.dependencies.matplotlib import pyplot as plt
from pyomo.common.dependencies.scipy import stats
imports_available = (
def _add_scatter(x, y, color, columns, theta_star, label=None):
    ax = plt.gca()
    xvar, yvar, loc = _get_variables(ax, columns)
    ax.scatter(theta_star[xvar], theta_star[yvar], c=color, s=35)