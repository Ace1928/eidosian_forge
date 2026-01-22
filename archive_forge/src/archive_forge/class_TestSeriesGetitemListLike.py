from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
from pandas.tseries.offsets import BDay
class TestSeriesGetitemListLike:

    @pytest.mark.parametrize('box', [list, np.array, Index, Series])
    def test_getitem_no_matches(self, box):
        ser = Series(['A', 'B'])
        key = Series(['C'], dtype=object)
        key = box(key)
        msg = "None of \\[Index\\(\\['C'\\], dtype='object|string'\\)\\] are in the \\[index\\]"
        with pytest.raises(KeyError, match=msg):
            ser[key]

    def test_getitem_intlist_intindex_periodvalues(self):
        ser = Series(period_range('2000-01-01', periods=10, freq='D'))
        result = ser[[2, 4]]
        exp = Series([pd.Period('2000-01-03', freq='D'), pd.Period('2000-01-05', freq='D')], index=[2, 4], dtype='Period[D]')
        tm.assert_series_equal(result, exp)
        assert result.dtype == 'Period[D]'

    @pytest.mark.parametrize('box', [list, np.array, Index])
    def test_getitem_intlist_intervalindex_non_int(self, box):
        dti = date_range('2000-01-03', periods=3)._with_freq(None)
        ii = pd.IntervalIndex.from_breaks(dti)
        ser = Series(range(len(ii)), index=ii)
        expected = ser.iloc[:1]
        key = box([0])
        msg = 'Series.__getitem__ treating keys as positions is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = ser[key]
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('box', [list, np.array, Index])
    @pytest.mark.parametrize('dtype', [np.int64, np.float64, np.uint64])
    def test_getitem_intlist_multiindex_numeric_level(self, dtype, box):
        idx = Index(range(4)).astype(dtype)
        dti = date_range('2000-01-03', periods=3)
        mi = pd.MultiIndex.from_product([idx, dti])
        ser = Series(range(len(mi))[::-1], index=mi)
        key = box([5])
        with pytest.raises(KeyError, match='5'):
            ser[key]

    def test_getitem_uint_array_key(self, any_unsigned_int_numpy_dtype):
        ser = Series([1, 2, 3])
        key = np.array([4], dtype=any_unsigned_int_numpy_dtype)
        with pytest.raises(KeyError, match='4'):
            ser[key]
        with pytest.raises(KeyError, match='4'):
            ser.loc[key]