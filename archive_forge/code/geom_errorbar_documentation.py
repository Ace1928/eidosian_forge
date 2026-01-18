from __future__ import annotations
import typing
import numpy as np
import pandas as pd
from .._utils import copy_missing_columns, resolution
from ..doctools import document
from .geom import geom
from .geom_path import geom_path
from .geom_segment import geom_segment

    Vertical interval represented as an errorbar

    {usage}

    Parameters
    ----------
    {common_parameters}
    width : float, default=0.5
        Bar width as a fraction of the resolution of the data.
    