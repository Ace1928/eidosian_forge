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
@pytest.mark.parametrize('conn', all_connectable)
@pytest.mark.parametrize('method', [None, 'multi'])
def test_to_sql(conn, method, test_frame1, request):
    if method == 'multi' and 'adbc' in conn:
        request.node.add_marker(pytest.mark.xfail(reason="'method' not implemented for ADBC drivers", strict=True))
    conn = request.getfixturevalue(conn)
    with pandasSQL_builder(conn, need_transaction=True) as pandasSQL:
        pandasSQL.to_sql(test_frame1, 'test_frame', method=method)
        assert pandasSQL.has_table('test_frame')
    assert count_rows(conn, 'test_frame') == len(test_frame1)