from __future__ import annotations
import typing
import numpy as np
import pandas as pd
from .._utils import SIZE_FACTOR, interleave, make_line_segments, to_rgba
from ..doctools import document
from .geom import geom
from .geom_path import geom_path

    Line segments

    {usage}

    Parameters
    ----------
    {common_parameters}
    lineend : Literal["butt", "round", "projecting"], default="butt"
        Line end style. This option is applied for solid linetypes.
    arrow : ~plotnine.geoms.geom_path.arrow, default=None
        Arrow specification. Default is no arrow.

    See Also
    --------
    plotnine.arrow : for adding arrowhead(s) to segments.
    