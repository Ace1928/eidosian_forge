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
def test_split_pattern_regex():
    arr = pa.array(['-foo---bar--', '---foo---b'])
    result = pc.split_pattern_regex(arr, pattern='-+')
    expected = pa.array([['', 'foo', 'bar', ''], ['', 'foo', 'b']])
    assert expected.equals(result)
    result = pc.split_pattern_regex(arr, '-+', max_splits=1)
    expected = pa.array([['', 'foo---bar--'], ['', 'foo---b']])
    assert expected.equals(result)
    with pytest.raises(NotImplementedError, match='Cannot split in reverse with regex'):
        result = pc.split_pattern_regex(arr, pattern='---', max_splits=1, reverse=True)