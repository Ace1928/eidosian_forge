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
def test_select_k_array():

    def validate_select_k(select_k_indices, arr, order, stable_sort=False):
        sorted_indices = pc.sort_indices(arr, sort_keys=[('dummy', order)])
        head_k_indices = sorted_indices.slice(0, len(select_k_indices))
        if stable_sort:
            assert select_k_indices == head_k_indices
        else:
            expected = pc.take(arr, head_k_indices)
            actual = pc.take(arr, select_k_indices)
            assert actual == expected
    arr = pa.array([1, 2, None, 0])
    for k in [0, 2, 4]:
        for order in ['descending', 'ascending']:
            result = pc.select_k_unstable(arr, k=k, sort_keys=[('dummy', order)])
            validate_select_k(result, arr, order)
        result = pc.top_k_unstable(arr, k=k)
        validate_select_k(result, arr, 'descending')
        result = pc.bottom_k_unstable(arr, k=k)
        validate_select_k(result, arr, 'ascending')
    result = pc.select_k_unstable(arr, options=pc.SelectKOptions(k=2, sort_keys=[('dummy', 'descending')]))
    validate_select_k(result, arr, 'descending')
    result = pc.select_k_unstable(arr, options=pc.SelectKOptions(k=2, sort_keys=[('dummy', 'ascending')]))
    validate_select_k(result, arr, 'ascending')
    assert pc.select_k_unstable(arr, 2, sort_keys=[('dummy', 'ascending')]) == result
    assert pc.select_k_unstable(arr, 2, [('dummy', 'ascending')]) == result