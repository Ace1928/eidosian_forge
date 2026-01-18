from __future__ import annotations
import typing
from contextlib import suppress
from warnings import warn
import numpy as np
import pandas as pd
from mizani.bounds import censor, expand_range_distinct, rescale, zero_range
from .._utils import match
from ..doctools import document
from ..exceptions import PlotnineError, PlotnineWarning
from ..iapi import range_view, scale_view
from ._expand import expand_range
from .range import RangeContinuous
from .scale import scale
def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
    """
        Transform dataframe
        """
    if len(df) == 0:
        return df
    aesthetics = set(self.aesthetics) & set(df.columns)
    for ae in aesthetics:
        with suppress(TypeError):
            df[ae] = self.transform(df[ae])
    return df