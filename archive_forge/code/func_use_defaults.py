from __future__ import annotations
import typing
from copy import deepcopy
import pandas as pd
from .._utils import (
from .._utils.registry import Register, Registry
from ..exceptions import PlotnineError
from ..layer import layer
from ..mapping import aes
from abc import ABC
def use_defaults(self, data: pd.DataFrame) -> pd.DataFrame:
    """
        Combine data with defaults and set aesthetics from parameters

        stats should not override this method.

        Parameters
        ----------
        data :
            Data used for drawing the geom.

        Returns
        -------
        out :
            Data used for drawing the geom.
        """
    missing = self.aesthetics() - set(self.aes_params.keys()) - set(data.columns)
    for ae in missing - self.REQUIRED_AES:
        if self.DEFAULT_AES[ae] is not None:
            data[ae] = self.DEFAULT_AES[ae]
    missing = self.aes_params.keys() - set(data.columns)
    for ae in self.aes_params:
        data[ae] = self.aes_params[ae]
    return data