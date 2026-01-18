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
@pytest.mark.parametrize('tiebreaker,expected_values', [('min', [3, 1, 4, 6, 4, 6, 1]), ('max', [3, 2, 5, 7, 5, 7, 2]), ('first', [3, 1, 4, 6, 5, 7, 2]), ('dense', [2, 1, 3, 4, 3, 4, 1])])
def test_rank_options_tiebreaker(tiebreaker, expected_values):
    arr = pa.array([1.2, 0.0, 5.3, None, 5.3, None, 0.0])
    rank_options = pc.RankOptions(sort_keys='ascending', null_placement='at_end', tiebreaker=tiebreaker)
    result = pc.rank(arr, options=rank_options)
    expected = pa.array(expected_values, type=pa.uint64())
    assert result.equals(expected)