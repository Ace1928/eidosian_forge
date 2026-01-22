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
def _get_daily_rule(self) -> str | None:
    ppd = periods_per_day(self._creso)
    days = self.deltas[0] / ppd
    if days % 7 == 0:
        wd = int_to_weekday[self.rep_stamp.weekday()]
        alias = f'W-{wd}'
        return _maybe_add_count(alias, days / 7)
    else:
        return _maybe_add_count('D', days)