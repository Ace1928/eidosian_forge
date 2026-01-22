import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
class LineCollection(Collection):
    """
    Represents a sequence of `.Line2D`\\s that should be drawn together.

    This class extends `.Collection` to represent a sequence of
    `.Line2D`\\s instead of just a sequence of `.Patch`\\s.
    Just as in `.Collection`, each property of a *LineCollection* may be either
    a single value or a list of values. This list is then used cyclically for
    each element of the LineCollection, so the property of the ``i``\\th element
    of the collection is::

      prop[i % len(prop)]

    The properties of each member of a *LineCollection* default to their values
    in :rc:`lines.*` instead of :rc:`patch.*`, and the property *colors* is
    added in place of *edgecolors*.
    """
    _edge_default = True

    def __init__(self, segments, *, zorder=2, **kwargs):
        """
        Parameters
        ----------
        segments : list of array-like
            A sequence (*line0*, *line1*, *line2*) of lines, where each line is a list
            of points::

                lineN = [(x0, y0), (x1, y1), ... (xm, ym)]

            or the equivalent Mx2 numpy array with two columns. Each line
            can have a different number of segments.
        linewidths : float or list of float, default: :rc:`lines.linewidth`
            The width of each line in points.
        colors : color or list of color, default: :rc:`lines.color`
            A sequence of RGBA tuples (e.g., arbitrary color strings, etc, not
            allowed).
        antialiaseds : bool or list of bool, default: :rc:`lines.antialiased`
            Whether to use antialiasing for each line.
        zorder : float, default: 2
            zorder of the lines once drawn.

        facecolors : color or list of color, default: 'none'
            When setting *facecolors*, each line is interpreted as a boundary
            for an area, implicitly closing the path from the last point to the
            first point. The enclosed area is filled with *facecolor*.
            In order to manually specify what should count as the "interior" of
            each line, please use `.PathCollection` instead, where the
            "interior" can be specified by appropriate usage of
            `~.path.Path.CLOSEPOLY`.

        **kwargs
            Forwarded to `.Collection`.
        """
        kwargs.setdefault('facecolors', 'none')
        super().__init__(zorder=zorder, **kwargs)
        self.set_segments(segments)

    def set_segments(self, segments):
        if segments is None:
            return
        self._paths = [mpath.Path(seg) if isinstance(seg, np.ma.MaskedArray) else mpath.Path(np.asarray(seg, float)) for seg in segments]
        self.stale = True
    set_verts = set_segments
    set_paths = set_segments

    def get_segments(self):
        """
        Returns
        -------
        list
            List of segments in the LineCollection. Each list item contains an
            array of vertices.
        """
        segments = []
        for path in self._paths:
            vertices = [vertex for vertex, _ in path.iter_segments(simplify=False)]
            vertices = np.asarray(vertices)
            segments.append(vertices)
        return segments

    def _get_default_linewidth(self):
        return mpl.rcParams['lines.linewidth']

    def _get_default_antialiased(self):
        return mpl.rcParams['lines.antialiased']

    def _get_default_edgecolor(self):
        return mpl.rcParams['lines.color']

    def _get_default_facecolor(self):
        return 'none'

    def set_alpha(self, alpha):
        super().set_alpha(alpha)
        if self._gapcolor is not None:
            self.set_gapcolor(self._original_gapcolor)

    def set_color(self, c):
        """
        Set the edgecolor(s) of the LineCollection.

        Parameters
        ----------
        c : color or list of colors
            Single color (all lines have same color), or a
            sequence of RGBA tuples; if it is a sequence the lines will
            cycle through the sequence.
        """
        self.set_edgecolor(c)
    set_colors = set_color

    def get_color(self):
        return self._edgecolors
    get_colors = get_color

    def set_gapcolor(self, gapcolor):
        """
        Set a color to fill the gaps in the dashed line style.

        .. note::

            Striped lines are created by drawing two interleaved dashed lines.
            There can be overlaps between those two, which may result in
            artifacts when using transparency.

            This functionality is experimental and may change.

        Parameters
        ----------
        gapcolor : color or list of colors or None
            The color with which to fill the gaps. If None, the gaps are
            unfilled.
        """
        self._original_gapcolor = gapcolor
        self._set_gapcolor(gapcolor)

    def _set_gapcolor(self, gapcolor):
        if gapcolor is not None:
            gapcolor = mcolors.to_rgba_array(gapcolor, self._alpha)
        self._gapcolor = gapcolor
        self.stale = True

    def get_gapcolor(self):
        return self._gapcolor

    def _get_inverse_paths_linestyles(self):
        """
        Returns the path and pattern for the gaps in the non-solid lines.

        This path and pattern is the inverse of the path and pattern used to
        construct the non-solid lines. For solid lines, we set the inverse path
        to nans to prevent drawing an inverse line.
        """
        path_patterns = [(mpath.Path(np.full((1, 2), np.nan)), ls) if ls == (0, None) else (path, mlines._get_inverse_dash_pattern(*ls)) for path, ls in zip(self._paths, itertools.cycle(self._linestyles))]
        return zip(*path_patterns)