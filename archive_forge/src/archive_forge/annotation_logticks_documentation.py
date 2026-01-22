from __future__ import annotations
import typing
import warnings
import numpy as np
import pandas as pd
from .._utils import log
from ..coords import coord_flip
from ..exceptions import PlotnineWarning
from ..scales.scale_continuous import scale_continuous as ScaleContinuous
from .annotate import annotate
from .geom_path import geom_path
from .geom_rug import geom_rug

        Calculate tick marks within a range

        Parameters
        ----------
        value_range: tuple
            Range for which to calculate ticks.

        base : number
            Base of logarithm

        Returns
        -------
        out: tuple
            (major, middle, minor) tick locations
        