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
def test_save_single_and_load_json(self):
    b = ArrayDataFrame([[6, 1], [2, 7]], 'c:int,a:long')
    path = os.path.join(self.tmpdir, 'a', 'b')
    makedirs(path, exist_ok=True)
    fa.save(b, path, format_hint='json', force_single=True)
    assert isfile(path)
    c = fa.load(path, format_hint='json', columns=['a', 'c'], as_fugue=True)
    df_eq(c, [[1, 6], [7, 2]], 'a:long,c:long', throw=True)
    b = ArrayDataFrame([[60, 1], [20, 7]], 'c:long,a:long')
    fa.save(b, path, format_hint='json', mode='overwrite')
    c = fa.load(path, format_hint='json', columns=['a', 'c'], as_fugue=True)
    df_eq(c, [[1, 60], [7, 20]], 'a:long,c:long', throw=True)