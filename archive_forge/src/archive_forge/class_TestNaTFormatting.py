from datetime import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p24p3
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.core.arrays import (
class TestNaTFormatting:

    def test_repr(self):
        assert repr(NaT) == 'NaT'

    def test_str(self):
        assert str(NaT) == 'NaT'

    def test_isoformat(self):
        assert NaT.isoformat() == 'NaT'