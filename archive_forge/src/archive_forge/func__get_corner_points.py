from numbers import Number
import copy
from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.figure_factory import utils
import plotly.graph_objects as go
def _get_corner_points(x0, y0, x1, y1):
    """
    Returns the corner points of a scatter rectangle

    :param x0: x-start
    :param y0: y-lower
    :param x1: x-end
    :param y1: y-upper
    :return: ([x], [y]), tuple of lists containing the x and y values
    """
    return ([x0, x1, x1, x0], [y0, y0, y1, y1])