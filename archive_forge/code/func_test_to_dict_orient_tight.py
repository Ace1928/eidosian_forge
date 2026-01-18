from collections import (
from datetime import datetime
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('index', [None, Index(['aa', 'bb']), Index(['aa', 'bb'], name='cc'), MultiIndex.from_tuples([('a', 'b'), ('a', 'c')]), MultiIndex.from_tuples([('a', 'b'), ('a', 'c')], names=['n1', 'n2'])])
@pytest.mark.parametrize('columns', [['x', 'y'], Index(['x', 'y']), Index(['x', 'y'], name='z'), MultiIndex.from_tuples([('x', 1), ('y', 2)]), MultiIndex.from_tuples([('x', 1), ('y', 2)], names=['z1', 'z2'])])
def test_to_dict_orient_tight(self, index, columns):
    df = DataFrame.from_records([[1, 3], [2, 4]], columns=columns, index=index)
    roundtrip = DataFrame.from_dict(df.to_dict(orient='tight'), orient='tight')
    tm.assert_frame_equal(df, roundtrip)