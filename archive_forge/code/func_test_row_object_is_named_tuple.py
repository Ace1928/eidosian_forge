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
def test_row_object_is_named_tuple(sqlite_engine):
    conn = sqlite_engine
    from sqlalchemy import Column, Integer, String
    from sqlalchemy.orm import declarative_base, sessionmaker
    BaseModel = declarative_base()

    class Test(BaseModel):
        __tablename__ = 'test_frame'
        id = Column(Integer, primary_key=True)
        string_column = Column(String(50))
    with conn.begin():
        BaseModel.metadata.create_all(conn)
    Session = sessionmaker(bind=conn)
    with Session() as session:
        df = DataFrame({'id': [0, 1], 'string_column': ['hello', 'world']})
        assert df.to_sql(name='test_frame', con=conn, index=False, if_exists='replace') == 2
        session.commit()
        test_query = session.query(Test.id, Test.string_column)
        df = DataFrame(test_query)
    assert list(df.columns) == ['id', 'string_column']