import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
class TestDropDuplicatesPeriodIndex(DropDuplicates):

    @pytest.fixture(params=['D', '3D', 'h', '2h', 'min', '2min', 's', '3s'])
    def freq(self, request):
        return request.param

    @pytest.fixture
    def idx(self, freq):
        return period_range('2011-01-01', periods=10, freq=freq, name='idx')