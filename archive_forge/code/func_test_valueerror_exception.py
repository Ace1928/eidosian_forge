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
def test_valueerror_exception(sqlite_engine):
    conn = sqlite_engine
    df = DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    with pytest.raises(ValueError, match='Empty table name specified'):
        df.to_sql(name='', con=conn, if_exists='replace', index=False)