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
def test_to_csv_multi_index(self):
    df = DataFrame([1], columns=pd.MultiIndex.from_arrays([[1], [2]]))
    exp_rows = [',1', ',2', '0,1']
    exp = tm.convert_rows_list_to_csv_str(exp_rows)
    assert df.to_csv() == exp
    exp_rows = ['1', '2', '1']
    exp = tm.convert_rows_list_to_csv_str(exp_rows)
    assert df.to_csv(index=False) == exp
    df = DataFrame([1], columns=pd.MultiIndex.from_arrays([[1], [2]]), index=pd.MultiIndex.from_arrays([[1], [2]]))
    exp_rows = [',,1', ',,2', '1,2,1']
    exp = tm.convert_rows_list_to_csv_str(exp_rows)
    assert df.to_csv() == exp
    exp_rows = ['1', '2', '1']
    exp = tm.convert_rows_list_to_csv_str(exp_rows)
    assert df.to_csv(index=False) == exp
    df = DataFrame([1], columns=pd.MultiIndex.from_arrays([['foo'], ['bar']]))
    exp_rows = [',foo', ',bar', '0,1']
    exp = tm.convert_rows_list_to_csv_str(exp_rows)
    assert df.to_csv() == exp
    exp_rows = ['foo', 'bar', '1']
    exp = tm.convert_rows_list_to_csv_str(exp_rows)
    assert df.to_csv(index=False) == exp