import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
class TestPeriodIndexNA(NATests):

    @pytest.fixture
    def index_without_na(self):
        return PeriodIndex(['2011-01-01', '2011-01-02'], freq='D')