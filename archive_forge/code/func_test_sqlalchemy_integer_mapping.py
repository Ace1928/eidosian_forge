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
@pytest.mark.parametrize('integer, expected', [('int8', 'SMALLINT'), ('Int8', 'SMALLINT'), ('uint8', 'SMALLINT'), ('UInt8', 'SMALLINT'), ('int16', 'SMALLINT'), ('Int16', 'SMALLINT'), ('uint16', 'INTEGER'), ('UInt16', 'INTEGER'), ('int32', 'INTEGER'), ('Int32', 'INTEGER'), ('uint32', 'BIGINT'), ('UInt32', 'BIGINT'), ('int64', 'BIGINT'), ('Int64', 'BIGINT'), (int, 'BIGINT' if np.dtype(int).name == 'int64' else 'INTEGER')])
def test_sqlalchemy_integer_mapping(conn, request, integer, expected):
    conn = request.getfixturevalue(conn)
    df = DataFrame([0, 1], columns=['a'], dtype=integer)
    with sql.SQLDatabase(conn) as db:
        table = sql.SQLTable('test_type', db, frame=df)
        result = str(table.table.c.a.type)
    assert result == expected