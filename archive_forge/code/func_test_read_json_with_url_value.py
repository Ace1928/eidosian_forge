import datetime
from datetime import timedelta
from decimal import Decimal
from io import (
import json
import os
import sys
import time
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import IS64
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
from pandas.io.json import ujson_dumps
@pytest.mark.parametrize('url', ['s3://example-fsspec/', 'gcs://another-fsspec/file.json', 'https://example-site.com/data', 'some-protocol://data.txt'])
def test_read_json_with_url_value(self, url):
    result = read_json(StringIO(f'{{"url":{{"0":"{url}"}}}}'))
    expected = DataFrame({'url': [url]})
    tm.assert_frame_equal(result, expected)