from collections import OrderedDict
from io import StringIO
import json
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.io.json._table_schema import (
@pytest.mark.parametrize('index_nm', [None, 'idx', 'index'])
@pytest.mark.parametrize('vals', [{'timedeltas': pd.timedelta_range('1h', periods=4, freq='min')}])
def test_read_json_table_orient_raises(self, index_nm, vals, recwarn):
    df = DataFrame(vals, index=pd.Index(range(4), name=index_nm))
    out = df.to_json(orient='table')
    with pytest.raises(NotImplementedError, match='can not yet read '):
        pd.read_json(out, orient='table')