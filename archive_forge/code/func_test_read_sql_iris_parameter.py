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
@pytest.mark.parametrize('conn', all_connectable_iris)
def test_read_sql_iris_parameter(conn, request, sql_strings):
    if 'adbc' in conn:
        request.node.add_marker(pytest.mark.xfail(reason="'params' not implemented for ADBC drivers", strict=True))
    conn_name = conn
    conn = request.getfixturevalue(conn)
    query = sql_strings['read_parameters'][flavor(conn_name)]
    params = ('Iris-setosa', 5.1)
    with pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction():
            iris_frame = pandasSQL.read_query(query, params=params)
    check_iris_frame(iris_frame)