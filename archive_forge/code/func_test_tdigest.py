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
def test_tdigest():
    arr = pa.array([1, 2, 3, 4])
    result = pc.tdigest(arr)
    assert result.to_pylist() == [2.5]
    arr = pa.chunked_array([pa.array([1, 2]), pa.array([3, 4])])
    result = pc.tdigest(arr)
    assert result.to_pylist() == [2.5]
    arr = pa.array([1, 2, 3, 4])
    result = pc.tdigest(arr, q=[0, 0.5, 1])
    assert result.to_pylist() == [1, 2.5, 4]
    arr = pa.chunked_array([pa.array([1, 2]), pa.array([3, 4])])
    result = pc.tdigest(arr, [0, 0.5, 1])
    assert result.to_pylist() == [1, 2.5, 4]