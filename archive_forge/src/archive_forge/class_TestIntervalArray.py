from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import IntervalDtype
from pandas import Interval
from pandas.core.arrays import IntervalArray
from pandas.tests.extension import base
class TestIntervalArray(base.ExtensionTests):
    divmod_exc = TypeError

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        return op_name in ['min', 'max']

    @pytest.mark.xfail(reason='Raises with incorrect message bc it disallows *all* listlikes instead of just wrong-length listlikes')
    def test_fillna_length_mismatch(self, data_missing):
        super().test_fillna_length_mismatch(data_missing)