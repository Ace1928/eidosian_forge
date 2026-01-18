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
def test_orc_reader_basic(dirpath):
    data = {'boolean1': np.array([False, True], dtype='bool'), 'byte1': np.array([1, 100], dtype='int8'), 'short1': np.array([1024, 2048], dtype='int16'), 'int1': np.array([65536, 65536], dtype='int32'), 'long1': np.array([9223372036854775807, 9223372036854775807], dtype='int64'), 'float1': np.array([1.0, 2.0], dtype='float32'), 'double1': np.array([-15.0, -5.0], dtype='float64'), 'bytes1': np.array([b'\x00\x01\x02\x03\x04', b''], dtype='object'), 'string1': np.array(['hi', 'bye'], dtype='object')}
    expected = pd.DataFrame.from_dict(data)
    inputfile = os.path.join(dirpath, 'TestOrcFile.test1.orc')
    got = read_orc(inputfile, columns=data.keys())
    tm.assert_equal(expected, got)