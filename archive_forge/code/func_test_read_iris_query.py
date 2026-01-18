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
def test_read_iris_query(conn, request):
    conn = request.getfixturevalue(conn)
    iris_frame = read_sql_query('SELECT * FROM iris', conn)
    check_iris_frame(iris_frame)
    iris_frame = pd.read_sql('SELECT * FROM iris', conn)
    check_iris_frame(iris_frame)
    iris_frame = pd.read_sql('SELECT * FROM iris where 0=1', conn)
    assert iris_frame.shape == (0, 5)
    assert 'SepalWidth' in iris_frame.columns