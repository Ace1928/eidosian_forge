from __future__ import annotations
import typing
from collections import Counter
from contextlib import suppress
from warnings import warn
import numpy as np
from .._utils import SIZE_FACTOR, make_line_segments, match, to_rgba
from ..doctools import document
from ..exceptions import PlotnineWarning
from .geom import geom

        Compute paths that create the arrow heads

        Parameters
        ----------
        x1, y1, x2, y2 : array_like
            List of points that define the tails of the arrows.
            The arrow heads will be at x1, y1. If you need them
            at x2, y2 reverse the input.
        panel_params : panel_view
            The scale information as may be required by the
            axes. At this point, that information is about
            ranges, ticks and labels. Attributes are of interest
            to the geom are:

            ```python
            "panel_params.x.range"  # tuple
            "panel_params.y.range"  # tuple
            ```
        coord : coord
            Coordinate (e.g. coord_cartesian) system of the geom.
        ax : axes
            Axes on which to plot.

        Returns
        -------
        out : list of Path
            Paths that create arrow heads
        