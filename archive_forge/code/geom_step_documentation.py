from __future__ import annotations
import typing
import numpy as np
import pandas as pd
from .._utils import copy_missing_columns
from ..doctools import document
from ..exceptions import PlotnineError
from .geom import geom
from .geom_path import geom_path

    Stepped connected points

    {usage}

    Parameters
    ----------
    {common_parameters}
    direction : Literal["hv", "vh", "mid"], default="hv"
        horizontal-vertical steps,
        vertical-horizontal steps or steps half-way between adjacent
        x values.

    See Also
    --------
    plotnine.geom_path : For documentation of extra parameters.
    