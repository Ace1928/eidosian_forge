from __future__ import annotations
import numpy as np
import pandas as pd
from xarray.coding.cftime_offsets import _MONTH_ABBREVIATIONS, _legacy_to_new_freq
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.common import _contains_datetime_like_objects
@property
def year_deltas(self):
    """Sorted unique year deltas."""
    if self._year_deltas is None:
        self._year_deltas = _unique_deltas(self.index.year)
    return self._year_deltas