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
@pytest.mark.network
@pytest.mark.single_cpu
def test_url_encoding_csv(httpserver, datapath):
    """
    read_csv should honor the requested encoding for URLs.

    GH 10424
    """
    with open(datapath('io', 'parser', 'data', 'unicode_series.csv'), 'rb') as f:
        httpserver.serve_content(content=f.read())
        df = read_csv(httpserver.url, encoding='latin-1', header=None)
    assert df.loc[15, 1] == 'Á köldum klaka (Cold Fever) (1994)'