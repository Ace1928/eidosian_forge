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
def test_orc_reader_date_high(dirpath):
    data = {'time': np.array(['2038-05-05 12:34:56.100000', '2038-05-05 12:34:56.100100', '2038-05-05 12:34:56.100200', '2038-05-05 12:34:56.100300', '2038-05-05 12:34:56.100400', '2038-05-05 12:34:56.100500', '2038-05-05 12:34:56.100600', '2038-05-05 12:34:56.100700', '2038-05-05 12:34:56.100800', '2038-05-05 12:34:56.100900'], dtype='datetime64[ns]'), 'date': np.array([datetime.date(2038, 12, 25), datetime.date(2038, 12, 25), datetime.date(2038, 12, 25), datetime.date(2038, 12, 25), datetime.date(2038, 12, 25), datetime.date(2038, 12, 25), datetime.date(2038, 12, 25), datetime.date(2038, 12, 25), datetime.date(2038, 12, 25), datetime.date(2038, 12, 25)], dtype='object')}
    expected = pd.DataFrame.from_dict(data)
    inputfile = os.path.join(dirpath, 'TestOrcFile.testDate2038.orc')
    got = read_orc(inputfile).iloc[:10]
    tm.assert_equal(expected, got)