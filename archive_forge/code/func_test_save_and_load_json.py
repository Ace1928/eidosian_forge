import copy
import os
import pickle
from datetime import datetime
from unittest import TestCase
import pandas as pd
import pytest
from pytest import raises
from triad.exceptions import InvalidOperationError
from triad.utils.io import isfile, makedirs, touch
import fugue.api as fa
import fugue.column.functions as ff
from fugue import (
from fugue.column import all_cols, col, lit
from fugue.dataframe.utils import _df_eq as df_eq
from fugue.execution.native_execution_engine import NativeExecutionEngine
def test_save_and_load_json(self):
    e = self.engine
    b = ArrayDataFrame([[6, 1], [3, 4], [2, 7], [4, 8], [6, 7]], 'c:int,a:long')
    path = os.path.join(self.tmpdir, 'a', 'b')
    fa.save(e.repartition(fa.as_fugue_engine_df(e, b), PartitionSpec(num=2)), path, format_hint='json')
    c = fa.load(path, format_hint='json', columns=['a', 'c'], as_fugue=True)
    df_eq(c, [[1, 6], [7, 2], [4, 3], [8, 4], [7, 6]], 'a:long,c:long', throw=True)