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
def test_read_json_from_to_json_results(self):
    df = DataFrame({'_id': {'row_0': 0}, 'category': {'row_0': 'Goods'}, 'recommender_id': {'row_0': 3}, 'recommender_name_jp': {'row_0': '浦田'}, 'recommender_name_en': {'row_0': 'Urata'}, 'name_jp': {'row_0': '博多人形(松尾吉将まつお よしまさ)'}, 'name_en': {'row_0': 'Hakata Dolls Matsuo'}})
    result1 = pd.read_json(StringIO(df.to_json()))
    result2 = DataFrame.from_dict(json.loads(df.to_json()))
    tm.assert_frame_equal(result1, df)
    tm.assert_frame_equal(result2, df)