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
def test_read_view_sqlite(sqlite_buildin):
    create_table = '\nCREATE TABLE groups (\n   group_id INTEGER,\n   name TEXT\n);\n'
    insert_into = "\nINSERT INTO groups VALUES\n    (1, 'name');\n"
    create_view = '\nCREATE VIEW group_view\nAS\nSELECT * FROM groups;\n'
    sqlite_buildin.execute(create_table)
    sqlite_buildin.execute(insert_into)
    sqlite_buildin.execute(create_view)
    result = pd.read_sql('SELECT * FROM group_view', sqlite_buildin)
    expected = DataFrame({'group_id': [1], 'name': 'name'})
    tm.assert_frame_equal(result, expected)