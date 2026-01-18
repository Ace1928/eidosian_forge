import datetime
from io import BytesIO
import re
import numpy as np
import pytest
from pandas import (
from pandas.tests.io.pytables.common import ensure_clean_store
from pandas.io.pytables import (
def test_unimplemented_dtypes_table_columns(setup_path):
    with ensure_clean_store(setup_path) as store:
        dtypes = [('date', datetime.date(2001, 1, 2))]
        for n, f in dtypes:
            df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=Index(list('ABCD'), dtype=object), index=Index([f'i-{i}' for i in range(30)], dtype=object))
            df[n] = f
            msg = re.escape(f'[{n}] is not implemented as a table column')
            with pytest.raises(TypeError, match=msg):
                store.append(f'df1_{n}', df)
    df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=Index(list('ABCD'), dtype=object), index=Index([f'i-{i}' for i in range(30)], dtype=object))
    df['obj1'] = 'foo'
    df['obj2'] = 'bar'
    df['datetime1'] = datetime.date(2001, 1, 2)
    df = df._consolidate()
    with ensure_clean_store(setup_path) as store:
        msg = re.escape('Cannot serialize the column [datetime1]\nbecause its data contents are not [string] but [date] object dtype')
        with pytest.raises(TypeError, match=msg):
            store.append('df_unimplemented', df)