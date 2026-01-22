import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
class EventCollection(LineCollection):
    """
    A collection of locations along a single axis at which an "event" occurred.

    The events are given by a 1-dimensional array. They do not have an
    amplitude and are displayed as parallel lines.
    """
    _edge_default = True

    def __init__(self, positions, orientation='horizontal', *, lineoffset=0, linelength=1, linewidth=None, color=None, linestyle='solid', antialiased=None, **kwargs):
        """
        Parameters
        ----------
        positions : 1D array-like
            Each value is an event.
        orientation : {'horizontal', 'vertical'}, default: 'horizontal'
            The sequence of events is plotted along this direction.
            The marker lines of the single events are along the orthogonal
            direction.
        lineoffset : float, default: 0
            The offset of the center of the markers from the origin, in the
            direction orthogonal to *orientation*.
        linelength : float, default: 1
            The total height of the marker (i.e. the marker stretches from
            ``lineoffset - linelength/2`` to ``lineoffset + linelength/2``).
        linewidth : float or list thereof, default: :rc:`lines.linewidth`
            The line width of the event lines, in points.
        color : color or list of colors, default: :rc:`lines.color`
            The color of the event lines.
        linestyle : str or tuple or list thereof, default: 'solid'
            Valid strings are ['solid', 'dashed', 'dashdot', 'dotted',
            '-', '--', '-.', ':']. Dash tuples should be of the form::

                (offset, onoffseq),

            where *onoffseq* is an even length tuple of on and off ink
            in points.
        antialiased : bool or list thereof, default: :rc:`lines.antialiased`
            Whether to use antialiasing for drawing the lines.
        **kwargs
            Forwarded to `.LineCollection`.

        Examples
        --------
        .. plot:: gallery/lines_bars_and_markers/eventcollection_demo.py
        """
        super().__init__([], linewidths=linewidth, linestyles=linestyle, colors=color, antialiaseds=antialiased, **kwargs)
        self._is_horizontal = True
        self._linelength = linelength
        self._lineoffset = lineoffset
        self.set_orientation(orientation)
        self.set_positions(positions)

    def get_positions(self):
        """
        Return an array containing the floating-point values of the positions.
        """
        pos = 0 if self.is_horizontal() else 1
        return [segment[0, pos] for segment in self.get_segments()]

    def set_positions(self, positions):
        """Set the positions of the events."""
        if positions is None:
            positions = []
        if np.ndim(positions) != 1:
            raise ValueError('positions must be one-dimensional')
        lineoffset = self.get_lineoffset()
        linelength = self.get_linelength()
        pos_idx = 0 if self.is_horizontal() else 1
        segments = np.empty((len(positions), 2, 2))
        segments[:, :, pos_idx] = np.sort(positions)[:, None]
        segments[:, 0, 1 - pos_idx] = lineoffset + linelength / 2
        segments[:, 1, 1 - pos_idx] = lineoffset - linelength / 2
        self.set_segments(segments)

    def add_positions(self, position):
        """Add one or more events at the specified positions."""
        if position is None or (hasattr(position, 'len') and len(position) == 0):
            return
        positions = self.get_positions()
        positions = np.hstack([positions, np.asanyarray(position)])
        self.set_positions(positions)
    extend_positions = append_positions = add_positions

    def is_horizontal(self):
        """True if the eventcollection is horizontal, False if vertical."""
        return self._is_horizontal

    def get_orientation(self):
        """
        Return the orientation of the event line ('horizontal' or 'vertical').
        """
        return 'horizontal' if self.is_horizontal() else 'vertical'

    def switch_orientation(self):
        """
        Switch the orientation of the event line, either from vertical to
        horizontal or vice versus.
        """
        segments = self.get_segments()
        for i, segment in enumerate(segments):
            segments[i] = np.fliplr(segment)
        self.set_segments(segments)
        self._is_horizontal = not self.is_horizontal()
        self.stale = True

    def set_orientation(self, orientation):
        """
        Set the orientation of the event line.

        Parameters
        ----------
        orientation : {'horizontal', 'vertical'}
        """
        is_horizontal = _api.check_getitem({'horizontal': True, 'vertical': False}, orientation=orientation)
        if is_horizontal == self.is_horizontal():
            return
        self.switch_orientation()

    def get_linelength(self):
        """Return the length of the lines used to mark each event."""
        return self._linelength

    def set_linelength(self, linelength):
        """Set the length of the lines used to mark each event."""
        if linelength == self.get_linelength():
            return
        lineoffset = self.get_lineoffset()
        segments = self.get_segments()
        pos = 1 if self.is_horizontal() else 0
        for segment in segments:
            segment[0, pos] = lineoffset + linelength / 2.0
            segment[1, pos] = lineoffset - linelength / 2.0
        self.set_segments(segments)
        self._linelength = linelength

    def get_lineoffset(self):
        """Return the offset of the lines used to mark each event."""
        return self._lineoffset

    def set_lineoffset(self, lineoffset):
        """Set the offset of the lines used to mark each event."""
        if lineoffset == self.get_lineoffset():
            return
        linelength = self.get_linelength()
        segments = self.get_segments()
        pos = 1 if self.is_horizontal() else 0
        for segment in segments:
            segment[0, pos] = lineoffset + linelength / 2.0
            segment[1, pos] = lineoffset - linelength / 2.0
        self.set_segments(segments)
        self._lineoffset = lineoffset

    def get_linewidth(self):
        """Get the width of the lines used to mark each event."""
        return super().get_linewidth()[0]

    def get_linewidths(self):
        return super().get_linewidth()

    def get_color(self):
        """Return the color of the lines used to mark each event."""
        return self.get_colors()[0]