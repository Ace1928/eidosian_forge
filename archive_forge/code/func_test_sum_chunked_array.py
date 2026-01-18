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
@pytest.mark.parametrize('arrow_type', numerical_arrow_types)
def test_sum_chunked_array(arrow_type):
    arr = pa.chunked_array([pa.array([1, 2, 3, 4], type=arrow_type)])
    assert pc.sum(arr).as_py() == 10
    arr = pa.chunked_array([pa.array([1, 2], type=arrow_type), pa.array([3, 4], type=arrow_type)])
    assert pc.sum(arr).as_py() == 10
    arr = pa.chunked_array([pa.array([1, 2], type=arrow_type), pa.array([], type=arrow_type), pa.array([3, 4], type=arrow_type)])
    assert pc.sum(arr).as_py() == 10
    arr = pa.chunked_array((), type=arrow_type)
    assert arr.num_chunks == 0
    assert pc.sum(arr).as_py() is None
    assert pc.sum(arr, min_count=0).as_py() == 0