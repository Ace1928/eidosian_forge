import gc
import decimal
import json
import multiprocessing as mp
import sys
import warnings
from collections import OrderedDict
from datetime import date, datetime, time, timedelta, timezone
import hypothesis as h
import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import pytest
from pyarrow.pandas_compat import get_logical_type, _pandas_api
from pyarrow.tests.util import invoke_script, random_ascii, rands
import pyarrow.tests.strategies as past
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
import pyarrow as pa
@pytest.mark.parametrize('columns', ([b'foo'], ['foo']))
def test_roundtrip_with_bytes_unicode(columns):
    if Version('2.0.0') <= Version(pd.__version__) < Version('2.3.0'):
        pytest.skip('Regression in pandas 2.0.0')
    df = pd.DataFrame(columns=columns)
    table1 = pa.Table.from_pandas(df)
    table2 = pa.Table.from_pandas(table1.to_pandas())
    assert table1.equals(table2)
    assert table1.schema.equals(table2.schema)
    assert table1.schema.metadata == table2.schema.metadata