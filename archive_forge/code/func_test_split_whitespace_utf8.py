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
def test_split_whitespace_utf8():
    arr = pa.array(['foo bar', ' foo  \u3000\tb'])
    result = pc.utf8_split_whitespace(arr)
    expected = pa.array([['foo', 'bar'], ['', 'foo', 'b']])
    assert expected.equals(result)
    result = pc.utf8_split_whitespace(arr, max_splits=1)
    expected = pa.array([['foo', 'bar'], ['', 'foo  \u3000\tb']])
    assert expected.equals(result)
    result = pc.utf8_split_whitespace(arr, max_splits=1, reverse=True)
    expected = pa.array([['foo', 'bar'], [' foo', 'b']])
    assert expected.equals(result)