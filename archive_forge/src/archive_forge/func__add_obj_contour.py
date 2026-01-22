import itertools
from pyomo.common.dependencies import (
from pyomo.common.dependencies.matplotlib import pyplot as plt
from pyomo.common.dependencies.scipy import stats
imports_available = (
def _add_obj_contour(x, y, color, columns, data, theta_star, label=None):
    ax = plt.gca()
    xvar, yvar, loc = _get_variables(ax, columns)
    try:
        X, Y, Z = _get_data_slice(xvar, yvar, columns, data, theta_star)
        triang = matplotlib.tri.Triangulation(X, Y)
        cmap = matplotlib.colormaps['Greys']
        plt.tricontourf(triang, Z, cmap=cmap)
    except:
        print('Objective contour plot for', xvar, yvar, 'slice failed')