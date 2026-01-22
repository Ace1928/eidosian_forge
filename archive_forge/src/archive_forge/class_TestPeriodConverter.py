from datetime import (
import subprocess
import sys
import numpy as np
import pytest
import pandas._config.config as cf
from pandas._libs.tslibs import to_offset
from pandas import (
import pandas._testing as tm
from pandas.plotting import (
from pandas.tseries.offsets import (
class TestPeriodConverter:

    @pytest.fixture
    def pc(self):
        return converter.PeriodConverter()

    @pytest.fixture
    def axis(self):

        class Axis:
            pass
        axis = Axis()
        axis.freq = 'D'
        return axis

    def test_convert_accepts_unicode(self, pc, axis):
        r1 = pc.convert('2012-1-1', None, axis)
        r2 = pc.convert('2012-1-1', None, axis)
        assert r1 == r2

    def test_conversion(self, pc, axis):
        rs = pc.convert(['2012-1-1'], None, axis)[0]
        xp = Period('2012-1-1').ordinal
        assert rs == xp
        rs = pc.convert('2012-1-1', None, axis)
        assert rs == xp
        rs = pc.convert([date(2012, 1, 1)], None, axis)[0]
        assert rs == xp
        rs = pc.convert(date(2012, 1, 1), None, axis)
        assert rs == xp
        rs = pc.convert([Timestamp('2012-1-1')], None, axis)[0]
        assert rs == xp
        rs = pc.convert(Timestamp('2012-1-1'), None, axis)
        assert rs == xp
        rs = pc.convert('2012-01-01', None, axis)
        assert rs == xp
        rs = pc.convert('2012-01-01 00:00:00+0000', None, axis)
        assert rs == xp
        rs = pc.convert(np.array(['2012-01-01 00:00:00', '2012-01-02 00:00:00'], dtype='datetime64[ns]'), None, axis)
        assert rs[0] == xp

    def test_integer_passthrough(self, pc, axis):
        rs = pc.convert([0, 1], None, axis)
        xp = [0, 1]
        assert rs == xp

    def test_convert_nested(self, pc, axis):
        data = ['2012-1-1', '2012-1-2']
        r1 = pc.convert([data, data], None, axis)
        r2 = [pc.convert(data, None, axis) for _ in range(2)]
        assert r1 == r2