from __future__ import annotations
import pickle
from datetime import timedelta
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import (
from xarray.tests import (
from xarray.tests.test_coding_times import (
@pytest.mark.parametrize(('string', 'expected'), list(ISO8601_LIKE_STRING_TESTS.values()), ids=list(ISO8601_LIKE_STRING_TESTS.keys()))
def test_parse_iso8601_like(string, expected):
    result = parse_iso8601_like(string)
    assert result == expected
    with pytest.raises(ValueError):
        parse_iso8601_like(string + '3')
        parse_iso8601_like(string + '.3')