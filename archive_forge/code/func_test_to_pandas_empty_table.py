from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
@pytest.mark.pandas
def test_to_pandas_empty_table():
    import pandas as pd
    import pandas.testing as tm
    df = pd.DataFrame({'a': [1, 2], 'b': [0.1, 0.2]})
    table = pa.table(df)
    result = table.schema.empty_table().to_pandas()
    assert result.shape == (0, 2)
    tm.assert_frame_equal(result, df.iloc[:0])