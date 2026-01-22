from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas._libs import lib
from pandas._libs.algos import unique_deltas
from pandas._libs.tslibs import (
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.dtypes import (
from pandas._libs.tslibs.fields import (
from pandas._libs.tslibs.offsets import (
from pandas._libs.tslibs.parsing import get_rule_month
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.common import is_numeric_dtype
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.algorithms import unique
class _FrequencyInferer:
    """
    Not sure if I can avoid the state machine here
    """

    def __init__(self, index) -> None:
        self.index = index
        self.i8values = index.asi8
        if isinstance(index, ABCIndex):
            self._creso = get_unit_from_dtype(index._data._ndarray.dtype)
        else:
            self._creso = get_unit_from_dtype(index._ndarray.dtype)
        if hasattr(index, 'tz'):
            if index.tz is not None:
                self.i8values = tz_convert_from_utc(self.i8values, index.tz, reso=self._creso)
        if len(index) < 3:
            raise ValueError('Need at least 3 dates to infer frequency')
        self.is_monotonic = self.index._is_monotonic_increasing or self.index._is_monotonic_decreasing

    @cache_readonly
    def deltas(self) -> npt.NDArray[np.int64]:
        return unique_deltas(self.i8values)

    @cache_readonly
    def deltas_asi8(self) -> npt.NDArray[np.int64]:
        return unique_deltas(self.index.asi8)

    @cache_readonly
    def is_unique(self) -> bool:
        return len(self.deltas) == 1

    @cache_readonly
    def is_unique_asi8(self) -> bool:
        return len(self.deltas_asi8) == 1

    def get_freq(self) -> str | None:
        """
        Find the appropriate frequency string to describe the inferred
        frequency of self.i8values

        Returns
        -------
        str or None
        """
        if not self.is_monotonic or not self.index._is_unique:
            return None
        delta = self.deltas[0]
        ppd = periods_per_day(self._creso)
        if delta and _is_multiple(delta, ppd):
            return self._infer_daily_rule()
        if self.hour_deltas in ([1, 17], [1, 65], [1, 17, 65]):
            return 'bh'
        if not self.is_unique_asi8:
            return None
        delta = self.deltas_asi8[0]
        pph = ppd // 24
        ppm = pph // 60
        pps = ppm // 60
        if _is_multiple(delta, pph):
            return _maybe_add_count('h', delta / pph)
        elif _is_multiple(delta, ppm):
            return _maybe_add_count('min', delta / ppm)
        elif _is_multiple(delta, pps):
            return _maybe_add_count('s', delta / pps)
        elif _is_multiple(delta, pps // 1000):
            return _maybe_add_count('ms', delta / (pps // 1000))
        elif _is_multiple(delta, pps // 1000000):
            return _maybe_add_count('us', delta / (pps // 1000000))
        else:
            return _maybe_add_count('ns', delta)

    @cache_readonly
    def day_deltas(self) -> list[int]:
        ppd = periods_per_day(self._creso)
        return [x / ppd for x in self.deltas]

    @cache_readonly
    def hour_deltas(self) -> list[int]:
        pph = periods_per_day(self._creso) // 24
        return [x / pph for x in self.deltas]

    @cache_readonly
    def fields(self) -> np.ndarray:
        return build_field_sarray(self.i8values, reso=self._creso)

    @cache_readonly
    def rep_stamp(self) -> Timestamp:
        return Timestamp(self.i8values[0], unit=self.index.unit)

    def month_position_check(self) -> str | None:
        return month_position_check(self.fields, self.index.dayofweek)

    @cache_readonly
    def mdiffs(self) -> npt.NDArray[np.int64]:
        nmonths = self.fields['Y'] * 12 + self.fields['M']
        return unique_deltas(nmonths.astype('i8'))

    @cache_readonly
    def ydiffs(self) -> npt.NDArray[np.int64]:
        return unique_deltas(self.fields['Y'].astype('i8'))

    def _infer_daily_rule(self) -> str | None:
        annual_rule = self._get_annual_rule()
        if annual_rule:
            nyears = self.ydiffs[0]
            month = MONTH_ALIASES[self.rep_stamp.month]
            alias = f'{annual_rule}-{month}'
            return _maybe_add_count(alias, nyears)
        quarterly_rule = self._get_quarterly_rule()
        if quarterly_rule:
            nquarters = self.mdiffs[0] / 3
            mod_dict = {0: 12, 2: 11, 1: 10}
            month = MONTH_ALIASES[mod_dict[self.rep_stamp.month % 3]]
            alias = f'{quarterly_rule}-{month}'
            return _maybe_add_count(alias, nquarters)
        monthly_rule = self._get_monthly_rule()
        if monthly_rule:
            return _maybe_add_count(monthly_rule, self.mdiffs[0])
        if self.is_unique:
            return self._get_daily_rule()
        if self._is_business_daily():
            return 'B'
        wom_rule = self._get_wom_rule()
        if wom_rule:
            return wom_rule
        return None

    def _get_daily_rule(self) -> str | None:
        ppd = periods_per_day(self._creso)
        days = self.deltas[0] / ppd
        if days % 7 == 0:
            wd = int_to_weekday[self.rep_stamp.weekday()]
            alias = f'W-{wd}'
            return _maybe_add_count(alias, days / 7)
        else:
            return _maybe_add_count('D', days)

    def _get_annual_rule(self) -> str | None:
        if len(self.ydiffs) > 1:
            return None
        if len(unique(self.fields['M'])) > 1:
            return None
        pos_check = self.month_position_check()
        if pos_check is None:
            return None
        else:
            return {'cs': 'YS', 'bs': 'BYS', 'ce': 'YE', 'be': 'BYE'}.get(pos_check)

    def _get_quarterly_rule(self) -> str | None:
        if len(self.mdiffs) > 1:
            return None
        if not self.mdiffs[0] % 3 == 0:
            return None
        pos_check = self.month_position_check()
        if pos_check is None:
            return None
        else:
            return {'cs': 'QS', 'bs': 'BQS', 'ce': 'QE', 'be': 'BQE'}.get(pos_check)

    def _get_monthly_rule(self) -> str | None:
        if len(self.mdiffs) > 1:
            return None
        pos_check = self.month_position_check()
        if pos_check is None:
            return None
        else:
            return {'cs': 'MS', 'bs': 'BMS', 'ce': 'ME', 'be': 'BME'}.get(pos_check)

    def _is_business_daily(self) -> bool:
        if self.day_deltas != [1, 3]:
            return False
        first_weekday = self.index[0].weekday()
        shifts = np.diff(self.i8values)
        ppd = periods_per_day(self._creso)
        shifts = np.floor_divide(shifts, ppd)
        weekdays = np.mod(first_weekday + np.cumsum(shifts), 7)
        return bool(np.all((weekdays == 0) & (shifts == 3) | (weekdays > 0) & (weekdays <= 4) & (shifts == 1)))

    def _get_wom_rule(self) -> str | None:
        weekdays = unique(self.index.weekday)
        if len(weekdays) > 1:
            return None
        week_of_months = unique((self.index.day - 1) // 7)
        week_of_months = week_of_months[week_of_months < 4]
        if len(week_of_months) == 0 or len(week_of_months) > 1:
            return None
        week = week_of_months[0] + 1
        wd = int_to_weekday[weekdays[0]]
        return f'WOM-{week}{wd}'