from __future__ import annotations
import contextlib
from contextlib import closing
import csv
from datetime import (
from io import StringIO
from pathlib import Path
import sqlite3
from typing import TYPE_CHECKING
import uuid
import numpy as np
import pytest
from pandas._libs import lib
from pandas.compat import (
from pandas.compat._optional import import_optional_dependency
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.util.version import Version
from pandas.io import sql
from pandas.io.sql import (
@pytest.mark.parametrize('conn', sqlalchemy_connectable)
@pytest.mark.parametrize('input', [{'foo': [np.inf]}, {'foo': [-np.inf]}, {'foo': [-np.inf], 'infe0': ['bar']}])
def test_to_sql_with_negative_npinf(conn, request, input):
    df = DataFrame(input)
    conn_name = conn
    conn = request.getfixturevalue(conn)
    if 'mysql' in conn_name:
        pymysql = pytest.importorskip('pymysql')
        if Version(pymysql.__version__) < Version('1.0.3') and 'infe0' in df.columns:
            mark = pytest.mark.xfail(reason='GH 36465')
            request.applymarker(mark)
        msg = 'inf cannot be used with MySQL'
        with pytest.raises(ValueError, match=msg):
            df.to_sql(name='foobar', con=conn, index=False)
    else:
        assert df.to_sql(name='foobar', con=conn, index=False) == 1
        res = sql.read_sql_table('foobar', conn)
        tm.assert_equal(df, res)