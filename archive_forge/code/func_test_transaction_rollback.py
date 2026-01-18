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
def test_transaction_rollback(conn, request):
    conn_name = conn
    conn = request.getfixturevalue(conn)
    with pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction() as trans:
            stmt = 'CREATE TABLE test_trans (A INT, B TEXT)'
            if 'adbc' in conn_name or isinstance(pandasSQL, SQLiteDatabase):
                trans.execute(stmt)
            else:
                from sqlalchemy import text
                stmt = text(stmt)
                trans.execute(stmt)

        class DummyException(Exception):
            pass
        ins_sql = "INSERT INTO test_trans (A,B) VALUES (1, 'blah')"
        if isinstance(pandasSQL, SQLDatabase):
            from sqlalchemy import text
            ins_sql = text(ins_sql)
        try:
            with pandasSQL.run_transaction() as trans:
                trans.execute(ins_sql)
                raise DummyException('error')
        except DummyException:
            pass
        with pandasSQL.run_transaction():
            res = pandasSQL.read_query('SELECT * FROM test_trans')
        assert len(res) == 0
        with pandasSQL.run_transaction() as trans:
            trans.execute(ins_sql)
            res2 = pandasSQL.read_query('SELECT * FROM test_trans')
        assert len(res2) == 1