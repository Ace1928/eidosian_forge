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
class scale_x_discrete(scale_position_discrete):
    """
    Discrete x position

    Parameters
    ----------
    {superclass_parameters}
    """
    _aesthetics = ['x', 'xmin', 'xmax', 'xend']