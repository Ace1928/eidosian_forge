from __future__ import annotations
import numpy as np
import pandas as pd
from xarray.coding.cftime_offsets import _MONTH_ABBREVIATIONS, _legacy_to_new_freq
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.common import _contains_datetime_like_objects
def _infer_daily_rule(self):
    annual_rule = self._get_annual_rule()
    if annual_rule:
        nyears = self.year_deltas[0]
        month = _MONTH_ABBREVIATIONS[self.index[0].month]
        alias = f'{annual_rule}-{month}'
        return _maybe_add_count(alias, nyears)
    quartely_rule = self._get_quartely_rule()
    if quartely_rule:
        nquarters = self.month_deltas[0] / 3
        mod_dict = {0: 12, 2: 11, 1: 10}
        month = _MONTH_ABBREVIATIONS[mod_dict[self.index[0].month % 3]]
        alias = f'{quartely_rule}-{month}'
        return _maybe_add_count(alias, nquarters)
    monthly_rule = self._get_monthly_rule()
    if monthly_rule:
        return _maybe_add_count(monthly_rule, self.month_deltas[0])
    if len(self.deltas) == 1:
        days = self.deltas[0] / _ONE_DAY
        return _maybe_add_count('D', days)
    return None