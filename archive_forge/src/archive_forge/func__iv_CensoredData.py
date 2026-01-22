from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import warnings
import numpy as np
from scipy import special, interpolate, stats
from scipy.stats._censored_data import CensoredData
from scipy.stats._common import ConfidenceInterval
def _iv_CensoredData(sample: npt.ArrayLike | CensoredData, param_name: str='sample') -> CensoredData:
    """Attempt to convert `sample` to `CensoredData`."""
    if not isinstance(sample, CensoredData):
        try:
            sample = CensoredData(uncensored=sample)
        except ValueError as e:
            message = str(e).replace('uncensored', param_name)
            raise type(e)(message) from e
    return sample