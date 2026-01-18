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
def test_options_sqlalchemy(conn, request, test_frame1):
    conn = request.getfixturevalue(conn)
    with pd.option_context('io.sql.engine', 'sqlalchemy'):
        with pandasSQL_builder(conn) as pandasSQL:
            with pandasSQL.run_transaction():
                assert pandasSQL.to_sql(test_frame1, 'test_frame1') == 4
                assert pandasSQL.has_table('test_frame1')
        num_entries = len(test_frame1)
        num_rows = count_rows(conn, 'test_frame1')
        assert num_rows == num_entries