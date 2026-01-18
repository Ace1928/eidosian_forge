import datetime
from io import BytesIO
import re
import numpy as np
import pytest
from pandas import (
from pandas.tests.io.pytables.common import ensure_clean_store
from pandas.io.pytables import (
@pytest.mark.parametrize('idx', [date_range('2019', freq='D', periods=3, tz='UTC'), CategoricalIndex(list('abc'))])
def test_to_hdf_multiindex_extension_dtype(idx, tmp_path, setup_path):
    mi = MultiIndex.from_arrays([idx, idx])
    df = DataFrame(0, index=mi, columns=['a'])
    path = tmp_path / setup_path
    with pytest.raises(NotImplementedError, match='Saving a MultiIndex'):
        df.to_hdf(path, key='df')