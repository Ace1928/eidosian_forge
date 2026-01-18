from __future__ import annotations
import typing
from abc import ABC
from copy import copy
from warnings import warn
import numpy as np
from .._utils import check_required_aesthetics, groupby_apply
from .._utils.registry import Register, Registry
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping.aes import X_AESTHETICS, Y_AESTHETICS
@staticmethod
def transform_position(data, trans_x: Optional[TransformCol]=None, trans_y: Optional[TransformCol]=None) -> pd.DataFrame:
    """
        Transform all the variables that map onto the x and y scales.

        Parameters
        ----------
        data : dataframe
            Data to transform
        trans_x : callable
            Transforms x scale mappings
            Takes one argument, either a scalar or an array-type
        trans_y : callable
            Transforms y scale mappings
            Takes one argument, either a scalar or an array-type
        """
    if len(data) == 0:
        return data
    if trans_x:
        xs = [name for name in data.columns if name in X_AESTHETICS]
        data[xs] = data[xs].apply(trans_x)
    if trans_y:
        ys = [name for name in data.columns if name in Y_AESTHETICS]
        data[ys] = data[ys].apply(trans_y)
    return data