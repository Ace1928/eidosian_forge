from numbers import Real
from matplotlib import _api
from matplotlib.axes import Axes
def _get_axes_aspect(ax):
    aspect = ax.get_aspect()
    if aspect == 'auto':
        aspect = 1.0
    return aspect