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
def test_read_json_orient_table_old_schema_version(self):
    df_json = '\n        {\n            "schema":{\n                "fields":[\n                    {"name":"index","type":"integer"},\n                    {"name":"a","type":"string"}\n                ],\n                "primaryKey":["index"],\n                "pandas_version":"0.20.0"\n            },\n            "data":[\n                {"index":0,"a":1},\n                {"index":1,"a":2.0},\n                {"index":2,"a":"s"}\n            ]\n        }\n        '
    expected = DataFrame({'a': [1, 2.0, 's']})
    result = pd.read_json(StringIO(df_json), orient='table')
    tm.assert_frame_equal(expected, result)