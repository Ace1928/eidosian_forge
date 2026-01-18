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
def test_warning_case_insensitive_table_name(conn, request, test_frame1):
    conn_name = conn
    if conn_name == 'sqlite_buildin' or 'adbc' in conn_name:
        request.applymarker(pytest.mark.xfail(reason='Does not raise warning'))
    conn = request.getfixturevalue(conn)
    with tm.assert_produces_warning(UserWarning, match="The provided table name 'TABLE1' is not found exactly as such in the database after writing the table, possibly due to case sensitivity issues. Consider using lower case table names."):
        with sql.SQLDatabase(conn) as db:
            db.check_case_sensitive('TABLE1', '')
    with tm.assert_produces_warning(None):
        test_frame1.to_sql(name='CaseSensitive', con=conn)