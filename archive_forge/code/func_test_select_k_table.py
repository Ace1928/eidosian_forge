from collections import namedtuple
import datetime
import decimal
from functools import lru_cache, partial
import inspect
import itertools
import math
import os
import pytest
import random
import sys
import textwrap
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import ArrowNotImplementedError
from pyarrow.tests import util
def test_select_k_table():

    def validate_select_k(select_k_indices, tbl, sort_keys, stable_sort=False):
        sorted_indices = pc.sort_indices(tbl, sort_keys=sort_keys)
        head_k_indices = sorted_indices.slice(0, len(select_k_indices))
        if stable_sort:
            assert select_k_indices == head_k_indices
        else:
            expected = pc.take(tbl, head_k_indices)
            actual = pc.take(tbl, select_k_indices)
            assert actual == expected
    table = pa.table({'a': [1, 2, 0], 'b': [1, 0, 1]})
    for k in [0, 2, 4]:
        result = pc.select_k_unstable(table, k=k, sort_keys=[('a', 'ascending')])
        validate_select_k(result, table, sort_keys=[('a', 'ascending')])
        result = pc.select_k_unstable(table, k=k, sort_keys=[(pc.field('a'), 'ascending'), ('b', 'ascending')])
        validate_select_k(result, table, sort_keys=[('a', 'ascending'), ('b', 'ascending')])
        result = pc.top_k_unstable(table, k=k, sort_keys=['a'])
        validate_select_k(result, table, sort_keys=[('a', 'descending')])
        result = pc.bottom_k_unstable(table, k=k, sort_keys=['a', 'b'])
        validate_select_k(result, table, sort_keys=[('a', 'ascending'), ('b', 'ascending')])
    with pytest.raises(ValueError, match="'select_k_unstable' cannot be called without options"):
        pc.select_k_unstable(table)
    with pytest.raises(ValueError, match='select_k_unstable requires a nonnegative `k`'):
        pc.select_k_unstable(table, k=-1, sort_keys=[('a', 'ascending')])
    with pytest.raises(ValueError, match='select_k_unstable requires a non-empty `sort_keys`'):
        pc.select_k_unstable(table, k=2, sort_keys=[])
    with pytest.raises(ValueError, match='not a valid sort order'):
        pc.select_k_unstable(table, k=k, sort_keys=[('a', 'nonscending')])
    with pytest.raises(ValueError, match='Invalid sort key column: No match for.*unknown'):
        pc.select_k_unstable(table, k=k, sort_keys=[('unknown', 'ascending')])