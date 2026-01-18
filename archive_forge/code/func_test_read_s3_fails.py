from io import BytesIO
import logging
import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import DataFrame
import pandas._testing as tm
from pandas.io.feather_format import read_feather
from pandas.io.parsers import read_csv
def test_read_s3_fails(self, s3so):
    msg = 'The specified bucket does not exist'
    with pytest.raises(OSError, match=msg):
        read_csv('s3://nyqpug/asdf.csv', storage_options=s3so)