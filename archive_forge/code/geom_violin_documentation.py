from __future__ import annotations
import typing
import numpy as np
import pandas as pd
from .._utils import groupby_apply, interleave, resolution
from ..doctools import document
from .geom import geom
from .geom_path import geom_path
from .geom_polygon import geom_polygon

    Return a dataframe with info needed to draw quantile segments
    