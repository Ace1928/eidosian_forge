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
@pytest.mark.parametrize('indent', [1, 2, 4])
def test_to_json_indent(self, indent):
    df = DataFrame([['foo', 'bar'], ['baz', 'qux']], columns=['a', 'b'])
    result = df.to_json(indent=indent)
    spaces = ' ' * indent
    expected = f'{{\n{spaces}"a":{{\n{spaces}{spaces}"0":"foo",\n{spaces}{spaces}"1":"baz"\n{spaces}}},\n{spaces}"b":{{\n{spaces}{spaces}"0":"bar",\n{spaces}{spaces}"1":"qux"\n{spaces}}}\n}}'
    assert result == expected