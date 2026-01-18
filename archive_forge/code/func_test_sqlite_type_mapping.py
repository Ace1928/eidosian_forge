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
def test_sqlite_type_mapping(sqlite_buildin):
    conn = sqlite_buildin
    df = DataFrame({'time': to_datetime(['2014-12-12 01:54', '2014-12-11 02:54'], utc=True)})
    db = sql.SQLiteDatabase(conn)
    table = sql.SQLiteTable('test_type', db, frame=df)
    schema = table.sql_schema()
    for col in schema.split('\n'):
        if col.split()[0].strip('"') == 'time':
            assert col.split()[1] == 'TIMESTAMP'