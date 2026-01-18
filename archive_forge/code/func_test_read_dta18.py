import bz2
import datetime as dt
from datetime import datetime
import gzip
import io
import os
import struct
import tarfile
import zipfile
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import CategoricalDtype
import pandas._testing as tm
from pandas.core.frame import (
from pandas.io.parsers import read_csv
from pandas.io.stata import (
def test_read_dta18(self, datapath):
    parsed_118 = self.read_dta(datapath('io', 'data', 'stata', 'stata14_118.dta'))
    parsed_118['Bytes'] = parsed_118['Bytes'].astype('O')
    expected = DataFrame.from_records([['Cat', 'Bogota', 'Bogotá', 1, 1.0, 'option b Ünicode', 1.0], ['Dog', 'Boston', 'Uzunköprü', np.nan, np.nan, np.nan, np.nan], ['Plane', 'Rome', 'Tromsø', 0, 0.0, 'option a', 0.0], ['Potato', 'Tokyo', 'Elâzığ', -4, 4.0, 4, 4], ['', '', '', 0, 0.3332999, 'option a', 1 / 3.0]], columns=['Things', 'Cities', 'Unicode_Cities_Strl', 'Ints', 'Floats', 'Bytes', 'Longs'])
    expected['Floats'] = expected['Floats'].astype(np.float32)
    for col in parsed_118.columns:
        tm.assert_almost_equal(parsed_118[col], expected[col])
    with StataReader(datapath('io', 'data', 'stata', 'stata14_118.dta')) as rdr:
        vl = rdr.variable_labels()
        vl_expected = {'Unicode_Cities_Strl': 'Here are some strls with Ünicode chars', 'Longs': 'long data', 'Things': 'Here are some things', 'Bytes': 'byte data', 'Ints': 'int data', 'Cities': 'Here are some cities', 'Floats': 'float data'}
        tm.assert_dict_equal(vl, vl_expected)
        assert rdr.data_label == 'This is a  Ünicode data label'