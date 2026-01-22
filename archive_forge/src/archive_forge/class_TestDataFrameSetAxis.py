import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
class TestDataFrameSetAxis(SharedSetAxisTests):

    @pytest.fixture
    def obj(self):
        df = DataFrame({'A': [1.1, 2.2, 3.3], 'B': [5.0, 6.1, 7.2], 'C': [4.4, 5.5, 6.6]}, index=[2010, 2011, 2012])
        return df