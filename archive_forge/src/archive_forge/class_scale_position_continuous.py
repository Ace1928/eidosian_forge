from __future__ import annotations
import typing
from itertools import chain
import numpy as np
import pandas as pd
from .._utils import array_kind, match
from .._utils.registry import alias
from ..doctools import document
from ..exceptions import PlotnineError
from ..iapi import range_view
from ._expand import expand_range
from .range import RangeContinuous
from .scale_continuous import scale_continuous
from .scale_datetime import scale_datetime
from .scale_discrete import scale_discrete
@document
class scale_position_continuous(scale_continuous):
    """
    Base class for continuous position scales

    Parameters
    ----------
    {superclass_parameters}
    """
    guide = None

    def map(self, x, limits=None):
        if not len(x):
            return x
        if limits is None:
            limits = self.limits
        scaled = self.oob(x, limits)
        scaled[pd.isna(scaled)] = self.na_value
        return scaled