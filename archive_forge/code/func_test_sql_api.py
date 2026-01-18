import datetime
import os
import pickle
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional
from unittest import TestCase
from uuid import uuid4
from triad.utils.io import write_text, join
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from fsspec.implementations.local import LocalFileSystem
from pytest import raises
from triad import SerializableRLock
import fugue.api as fa
from fugue import (
from fugue.column import col
from fugue.column import functions as ff
from fugue.column import lit
from fugue.dataframe.utils import _df_eq as df_eq
from fugue.exceptions import (
@pytest.mark.skipif(not HAS_QPD, reason='qpd not working')
def test_sql_api(self):

    def tr(df: pd.DataFrame, n=1) -> pd.DataFrame:
        return df + n
    with fa.engine_context(self.engine):
        df1 = fa.as_fugue_df([[0, 1], [2, 3], [4, 5]], schema='a:long,b:int')
        df2 = pd.DataFrame([[0, 10], [1, 100]], columns=['a', 'c'])
        sdf1 = fa.raw_sql('SELECT ', df1, '.a, b FROM ', df1, ' WHERE a<4')
        sdf2 = fa.raw_sql('SELECT * FROM ', df2, ' WHERE a<1')
        sdf3 = fa.fugue_sql('\n                SELECT sdf1.a,sdf1.b,c FROM sdf1 INNER JOIN sdf2 ON sdf1.a=sdf2.a\n                TRANSFORM USING tr SCHEMA *\n                ')
        res = fa.fugue_sql_flow('\n                TRANSFORM x USING tr(n=2) SCHEMA *\n                YIELD LOCAL DATAFRAME AS res\n                PRINT sdf1\n                ', x=sdf3).run()
        df_eq(res['res'], [[3, 4, 13]], schema='a:long,b:int,c:long', check_schema=False, throw=True)
        sdf4 = fa.fugue_sql('\n                SELECT sdf1.a,b,c FROM sdf1 INNER JOIN sdf2 ON sdf1.a=sdf2.a\n                TRANSFORM USING tr SCHEMA *\n                ', as_fugue=False, as_local=True)
        assert not isinstance(sdf4, DataFrame)
        assert fa.is_local(sdf4)