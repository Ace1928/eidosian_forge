from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_unsigned_integer_dtype
from pandas.core.dtypes.dtypes import IntervalDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
import pandas.core.common as com
class TestFromTuples(ConstructorTests):
    """Tests specific to IntervalIndex.from_tuples"""

    @pytest.fixture
    def constructor(self):
        return IntervalIndex.from_tuples

    def get_kwargs_from_breaks(self, breaks, closed='right'):
        """
        converts intervals in breaks format to a dictionary of kwargs to
        specific to the format expected by IntervalIndex.from_tuples
        """
        if is_unsigned_integer_dtype(breaks):
            pytest.skip(f'{breaks.dtype} not relevant IntervalIndex.from_tuples tests')
        if len(breaks) == 0:
            return {'data': breaks}
        tuples = list(zip(breaks[:-1], breaks[1:]))
        if isinstance(breaks, (list, tuple)):
            return {'data': tuples}
        elif isinstance(getattr(breaks, 'dtype', None), CategoricalDtype):
            return {'data': breaks._constructor(tuples)}
        return {'data': com.asarray_tuplesafe(tuples)}

    def test_constructor_errors(self):
        tuples = [(0, 1), 2, (3, 4)]
        msg = 'IntervalIndex.from_tuples received an invalid item, 2'
        with pytest.raises(TypeError, match=msg.format(t=tuples)):
            IntervalIndex.from_tuples(tuples)
        tuples = [(0, 1), (2,), (3, 4)]
        msg = 'IntervalIndex.from_tuples requires tuples of length 2, got {t}'
        with pytest.raises(ValueError, match=msg.format(t=tuples)):
            IntervalIndex.from_tuples(tuples)
        tuples = [(0, 1), (2, 3, 4), (5, 6)]
        with pytest.raises(ValueError, match=msg.format(t=tuples)):
            IntervalIndex.from_tuples(tuples)

    def test_na_tuples(self):
        na_tuple = [(0, 1), (np.nan, np.nan), (2, 3)]
        idx_na_tuple = IntervalIndex.from_tuples(na_tuple)
        idx_na_element = IntervalIndex.from_tuples([(0, 1), np.nan, (2, 3)])
        tm.assert_index_equal(idx_na_tuple, idx_na_element)