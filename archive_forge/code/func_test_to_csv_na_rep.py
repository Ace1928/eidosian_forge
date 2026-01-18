import io
import os
import sys
from zipfile import ZipFile
from _csv import Error
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_csv_na_rep(self):
    df = DataFrame({'a': [0, np.nan], 'b': [0, 1], 'c': [2, 3]})
    expected_rows = ['a,b,c', '0.0,0,2', '_,1,3']
    expected = tm.convert_rows_list_to_csv_str(expected_rows)
    assert df.set_index('a').to_csv(na_rep='_') == expected
    assert df.set_index(['a', 'b']).to_csv(na_rep='_') == expected
    df = DataFrame({'a': np.nan, 'b': [0, 1], 'c': [2, 3]})
    expected_rows = ['a,b,c', '_,0,2', '_,1,3']
    expected = tm.convert_rows_list_to_csv_str(expected_rows)
    assert df.set_index('a').to_csv(na_rep='_') == expected
    assert df.set_index(['a', 'b']).to_csv(na_rep='_') == expected
    df = DataFrame({'a': 0, 'b': [0, 1], 'c': [2, 3]})
    expected_rows = ['a,b,c', '0,0,2', '0,1,3']
    expected = tm.convert_rows_list_to_csv_str(expected_rows)
    assert df.set_index('a').to_csv(na_rep='_') == expected
    assert df.set_index(['a', 'b']).to_csv(na_rep='_') == expected
    csv = pd.Series(['a', pd.NA, 'c']).to_csv(na_rep='ZZZZZ')
    expected = tm.convert_rows_list_to_csv_str([',0', '0,a', '1,ZZZZZ', '2,c'])
    assert expected == csv