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
def test_xsqlite_execute_fail(sqlite_buildin):
    create_sql = '\n    CREATE TABLE test\n    (\n    a TEXT,\n    b TEXT,\n    c REAL,\n    PRIMARY KEY (a, b)\n    );\n    '
    cur = sqlite_buildin.cursor()
    cur.execute(create_sql)
    with sql.pandasSQL_builder(sqlite_buildin) as pandas_sql:
        pandas_sql.execute('INSERT INTO test VALUES("foo", "bar", 1.234)')
        pandas_sql.execute('INSERT INTO test VALUES("foo", "baz", 2.567)')
        with pytest.raises(sql.DatabaseError, match='Execution failed on sql'):
            pandas_sql.execute('INSERT INTO test VALUES("foo", "bar", 7)')