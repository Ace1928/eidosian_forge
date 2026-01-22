import plotly.colors as clrs
from plotly.graph_objs import graph_objs as go
from plotly import exceptions
from plotly import optional_imports
from skimage import measure
def _is_invalid_contour(x, y):
    """
    Utility function for _contour_trace

    Contours with an area of the order as 1 pixel are considered spurious.
    """
    too_small = np.all(np.abs(x - x[0]) < 2) and np.all(np.abs(y - y[0]) < 2)
    return too_small