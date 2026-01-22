from contextlib import ExitStack
import copy
import itertools
from numbers import Integral, Number
from cycler import cycler
import numpy as np
import matplotlib as mpl
from . import (_api, _docstring, backend_tools, cbook, collections, colors,
from .lines import Line2D
from .patches import Circle, Rectangle, Ellipse, Polygon
from .transforms import TransformedPatchPath, Affine2D
class ToolLineHandles:
    """
    Control handles for canvas tools.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Matplotlib Axes where tool handles are displayed.
    positions : 1D array
        Positions of handles in data coordinates.
    direction : {"horizontal", "vertical"}
        Direction of handles, either 'vertical' or 'horizontal'
    line_props : dict, optional
        Additional line properties. See `.Line2D`.
    useblit : bool, default: True
        Whether to use blitting for faster drawing (if supported by the
        backend). See the tutorial :ref:`blitting`
        for details.
    """

    @_api.make_keyword_only('3.7', 'line_props')
    def __init__(self, ax, positions, direction, line_props=None, useblit=True):
        self.ax = ax
        _api.check_in_list(['horizontal', 'vertical'], direction=direction)
        self._direction = direction
        line_props = {**(line_props if line_props is not None else {}), 'visible': False, 'animated': useblit}
        line_fun = ax.axvline if self.direction == 'horizontal' else ax.axhline
        self._artists = [line_fun(p, **line_props) for p in positions]

    @property
    def artists(self):
        return tuple(self._artists)

    @property
    def positions(self):
        """Positions of the handle in data coordinates."""
        method = 'get_xdata' if self.direction == 'horizontal' else 'get_ydata'
        return [getattr(line, method)()[0] for line in self.artists]

    @property
    def direction(self):
        """Direction of the handle: 'vertical' or 'horizontal'."""
        return self._direction

    def set_data(self, positions):
        """
        Set x- or y-positions of handles, depending on if the lines are
        vertical or horizontal.

        Parameters
        ----------
        positions : tuple of length 2
            Set the positions of the handle in data coordinates
        """
        method = 'set_xdata' if self.direction == 'horizontal' else 'set_ydata'
        for line, p in zip(self.artists, positions):
            getattr(line, method)([p, p])

    def set_visible(self, value):
        """Set the visibility state of the handles artist."""
        for artist in self.artists:
            artist.set_visible(value)

    def set_animated(self, value):
        """Set the animated state of the handles artist."""
        for artist in self.artists:
            artist.set_animated(value)

    def remove(self):
        """Remove the handles artist from the figure."""
        for artist in self._artists:
            artist.remove()

    def closest(self, x, y):
        """
        Return index and pixel distance to closest handle.

        Parameters
        ----------
        x, y : float
            x, y position from which the distance will be calculated to
            determinate the closest handle

        Returns
        -------
        index, distance : index of the handle and its distance from
            position x, y
        """
        if self.direction == 'horizontal':
            p_pts = np.array([self.ax.transData.transform((p, 0))[0] for p in self.positions])
            dist = abs(p_pts - x)
        else:
            p_pts = np.array([self.ax.transData.transform((0, p))[1] for p in self.positions])
            dist = abs(p_pts - y)
        index = np.argmin(dist)
        return (index, dist[index])