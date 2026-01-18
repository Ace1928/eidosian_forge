from __future__ import annotations
from itertools import product
from typing import Callable, Literal
import numpy as np
import pandas as pd
import pytest
from xarray import CFTimeIndex
from xarray.coding.cftime_offsets import (
from xarray.coding.frequencies import infer_freq
from xarray.core.dataarray import DataArray
from xarray.tests import (
@pytest.mark.parametrize(('offset', 'expected'), [(BaseCFTimeOffset(), None), (MonthBegin(), 'MS'), (MonthEnd(), 'ME'), (YearBegin(), 'YS-JAN'), (YearEnd(), 'YE-DEC'), (QuarterBegin(), 'QS-MAR'), (QuarterEnd(), 'QE-MAR'), (Day(), 'D'), (Hour(), 'h'), (Minute(), 'min'), (Second(), 's'), (Millisecond(), 'ms'), (Microsecond(), 'us')], ids=_id_func)
def test_rule_code(offset, expected):
    assert offset.rule_code() == expected