import datetime
from decimal import Decimal
from io import BytesIO
import os
import pathlib
import numpy as np
import pytest
import pandas as pd
from pandas import read_orc
import pandas._testing as tm
from pandas.core.arrays import StringArray
import pyarrow as pa
@pytest.mark.parametrize('index', [pd.RangeIndex(start=2, stop=5, step=1), pd.RangeIndex(start=0, stop=3, step=1, name='non-default'), pd.Index([1, 2, 3])])
def test_to_orc_non_default_index(index):
    df = pd.DataFrame({'a': [1, 2, 3]}, index=index)
    msg = 'orc does not support serializing a non-default index|orc does not serialize index meta-data'
    with pytest.raises(ValueError, match=msg):
        df.to_orc()