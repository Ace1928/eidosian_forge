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
@pytest.fixture
def types_data_frame(types_data):
    dtypes = {'TextCol': 'str', 'DateCol': 'str', 'IntDateCol': 'int64', 'IntDateOnlyCol': 'int64', 'FloatCol': 'float', 'IntCol': 'int64', 'BoolCol': 'int64', 'IntColWithNull': 'float', 'BoolColWithNull': 'float'}
    df = DataFrame(types_data)
    return df[dtypes.keys()].astype(dtypes)