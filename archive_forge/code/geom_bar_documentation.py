from __future__ import annotations
import typing
from .._utils import resolution
from ..doctools import document
from .geom_rect import geom_rect

    Bar plot

    {usage}

    Parameters
    ----------
    {common_parameters}
    width : float, default=None
        Bar width. If `None`{.py}, the width is set to
        `90%` of the resolution of the data.

    See Also
    --------
    plotnine.geom_histogram
    