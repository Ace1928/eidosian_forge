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
def test_to_json_float_index(self):
    data = pd.Series(1, index=[1.0, 2.0])
    result = data.to_json(orient='table', date_format='iso')
    result = json.loads(result, object_pairs_hook=OrderedDict)
    result['schema'].pop('pandas_version')
    expected = OrderedDict([('schema', {'fields': [{'name': 'index', 'type': 'number'}, {'name': 'values', 'type': 'integer'}], 'primaryKey': ['index']}), ('data', [OrderedDict([('index', 1.0), ('values', 1)]), OrderedDict([('index', 2.0), ('values', 1)])])])
    assert result == expected