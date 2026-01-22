import functools
import itertools
import logging
import math
from numbers import Integral, Number, Real
import numpy as np
from numpy import ma
import matplotlib as mpl
import matplotlib.category  # Register category unit converter as side effect.
import matplotlib.cbook as cbook
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.contour as mcontour
import matplotlib.dates  # noqa # Register date unit converter as side effect.
import matplotlib.image as mimage
import matplotlib.legend as mlegend
import matplotlib.lines as mlines
import matplotlib.markers as mmarkers
import matplotlib.mlab as mlab
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.quiver as mquiver
import matplotlib.stackplot as mstack
import matplotlib.streamplot as mstream
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.tri as mtri
import matplotlib.units as munits
from matplotlib import _api, _docstring, _preprocess_data
from matplotlib.axes._base import (
from matplotlib.axes._secondary_axes import SecondaryAxis
from matplotlib.container import BarContainer, ErrorbarContainer, StemContainer
@staticmethod
def _parse_scatter_color_args(c, edgecolors, kwargs, xsize, get_next_color_func):
    """
        Helper function to process color related arguments of `.Axes.scatter`.

        Argument precedence for facecolors:

        - c (if not None)
        - kwargs['facecolor']
        - kwargs['facecolors']
        - kwargs['color'] (==kwcolor)
        - 'b' if in classic mode else the result of ``get_next_color_func()``

        Argument precedence for edgecolors:

        - kwargs['edgecolor']
        - edgecolors (is an explicit kw argument in scatter())
        - kwargs['color'] (==kwcolor)
        - 'face' if not in classic mode else None

        Parameters
        ----------
        c : color or sequence or sequence of color or None
            See argument description of `.Axes.scatter`.
        edgecolors : color or sequence of color or {'face', 'none'} or None
            See argument description of `.Axes.scatter`.
        kwargs : dict
            Additional kwargs. If these keys exist, we pop and process them:
            'facecolors', 'facecolor', 'edgecolor', 'color'
            Note: The dict is modified by this function.
        xsize : int
            The size of the x and y arrays passed to `.Axes.scatter`.
        get_next_color_func : callable
            A callable that returns a color. This color is used as facecolor
            if no other color is provided.

            Note, that this is a function rather than a fixed color value to
            support conditional evaluation of the next color.  As of the
            current implementation obtaining the next color from the
            property cycle advances the cycle. This must only happen if we
            actually use the color, which will only be decided within this
            method.

        Returns
        -------
        c
            The input *c* if it was not *None*, else a color derived from the
            other inputs or defaults.
        colors : array(N, 4) or None
            The facecolors as RGBA values, or *None* if a colormap is used.
        edgecolors
            The edgecolor.

        """
    facecolors = kwargs.pop('facecolors', None)
    facecolors = kwargs.pop('facecolor', facecolors)
    edgecolors = kwargs.pop('edgecolor', edgecolors)
    kwcolor = kwargs.pop('color', None)
    if kwcolor is not None and c is not None:
        raise ValueError("Supply a 'c' argument or a 'color' kwarg but not both; they differ but their functionalities overlap.")
    if kwcolor is not None:
        try:
            mcolors.to_rgba_array(kwcolor)
        except ValueError as err:
            raise ValueError("'color' kwarg must be a color or sequence of color specs.  For a sequence of values to be color-mapped, use the 'c' argument instead.") from err
        if edgecolors is None:
            edgecolors = kwcolor
        if facecolors is None:
            facecolors = kwcolor
    if edgecolors is None and (not mpl.rcParams['_internal.classic_mode']):
        edgecolors = mpl.rcParams['scatter.edgecolors']
    c_was_none = c is None
    if c is None:
        c = facecolors if facecolors is not None else 'b' if mpl.rcParams['_internal.classic_mode'] else get_next_color_func()
    c_is_string_or_strings = isinstance(c, str) or (np.iterable(c) and len(c) > 0 and isinstance(cbook._safe_first_finite(c), str))

    def invalid_shape_exception(csize, xsize):
        return ValueError(f"'c' argument has {csize} elements, which is inconsistent with 'x' and 'y' with size {xsize}.")
    c_is_mapped = False
    valid_shape = True
    if not c_was_none and kwcolor is None and (not c_is_string_or_strings):
        try:
            c = np.asanyarray(c, dtype=float)
        except ValueError:
            pass
        else:
            if c.shape == (1, 4) or c.shape == (1, 3):
                c_is_mapped = False
                if c.size != xsize:
                    valid_shape = False
            elif c.size == xsize:
                c = c.ravel()
                c_is_mapped = True
            else:
                if c.shape in ((3,), (4,)):
                    _api.warn_external('*c* argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with *x* & *y*.  Please use the *color* keyword-argument or provide a 2D array with a single row if you intend to specify the same RGB or RGBA value for all points.')
                valid_shape = False
    if not c_is_mapped:
        try:
            colors = mcolors.to_rgba_array(c)
        except (TypeError, ValueError) as err:
            if 'RGBA values should be within 0-1 range' in str(err):
                raise
            else:
                if not valid_shape:
                    raise invalid_shape_exception(c.size, xsize) from err
                raise ValueError(f"'c' argument must be a color, a sequence of colors, or a sequence of numbers, not {c!r}") from err
        else:
            if len(colors) not in (0, 1, xsize):
                raise invalid_shape_exception(len(colors), xsize)
    else:
        colors = None
    return (c, colors, edgecolors)