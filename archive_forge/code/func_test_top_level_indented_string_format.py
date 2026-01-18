from collections.abc import Iterable
import datetime
import decimal
import hypothesis as h
import hypothesis.strategies as st
import itertools
import pytest
import struct
import subprocess
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
def test_top_level_indented_string_format():
    arr = pa.array(['', None, 'foo'])
    result = arr.to_string(top_level_indent=1)
    expected = ' [\n   "",\n   null,\n   "foo"\n ]'
    assert result == expected