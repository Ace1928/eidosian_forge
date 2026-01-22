from __future__ import annotations
import numpy as np
import pandas as pd
from xarray.coding.cftime_offsets import _MONTH_ABBREVIATIONS, _legacy_to_new_freq
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.common import _contains_datetime_like_objects
class _CFTimeFrequencyInferer:

    def __init__(self, index):
        self.index = index
        self.values = index.asi8
        if len(index) < 3:
            raise ValueError('Need at least 3 dates to infer frequency')
        self.is_monotonic = self.index.is_monotonic_decreasing or self.index.is_monotonic_increasing
        self._deltas = None
        self._year_deltas = None
        self._month_deltas = None

    def get_freq(self):
        """Find the appropriate frequency string to describe the inferred frequency of self.index

        Adapted from `pandas.tsseries.frequencies._FrequencyInferer.get_freq` for CFTimeIndexes.

        Returns
        -------
        str or None
        """
        if not self.is_monotonic or not self.index.is_unique:
            return None
        delta = self.deltas[0]
        if _is_multiple(delta, _ONE_DAY):
            return self._infer_daily_rule()
        elif not len(self.deltas) == 1:
            return None
        if _is_multiple(delta, _ONE_HOUR):
            return _maybe_add_count('h', delta / _ONE_HOUR)
        elif _is_multiple(delta, _ONE_MINUTE):
            return _maybe_add_count('min', delta / _ONE_MINUTE)
        elif _is_multiple(delta, _ONE_SECOND):
            return _maybe_add_count('s', delta / _ONE_SECOND)
        elif _is_multiple(delta, _ONE_MILLI):
            return _maybe_add_count('ms', delta / _ONE_MILLI)
        else:
            return _maybe_add_count('us', delta / _ONE_MICRO)

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

    def _get_annual_rule(self):
        if len(self.year_deltas) > 1:
            return None
        if len(np.unique(self.index.month)) > 1:
            return None
        return {'cs': 'YS', 'ce': 'YE'}.get(month_anchor_check(self.index))

    def _get_quartely_rule(self):
        if len(self.month_deltas) > 1:
            return None
        if self.month_deltas[0] % 3 != 0:
            return None
        return {'cs': 'QS', 'ce': 'QE'}.get(month_anchor_check(self.index))

    def _get_monthly_rule(self):
        if len(self.month_deltas) > 1:
            return None
        return {'cs': 'MS', 'ce': 'ME'}.get(month_anchor_check(self.index))

    @property
    def deltas(self):
        """Sorted unique timedeltas as microseconds."""
        if self._deltas is None:
            self._deltas = _unique_deltas(self.values)
        return self._deltas

    @property
    def year_deltas(self):
        """Sorted unique year deltas."""
        if self._year_deltas is None:
            self._year_deltas = _unique_deltas(self.index.year)
        return self._year_deltas

    @property
    def month_deltas(self):
        """Sorted unique month deltas."""
        if self._month_deltas is None:
            self._month_deltas = _unique_deltas(self.index.year * 12 + self.index.month)
        return self._month_deltas