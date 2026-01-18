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
def test_unique_simple():
    cases = [(pa.array([1, 2, 3, 1, 2, 3]), pa.array([1, 2, 3])), (pa.array(['foo', None, 'bar', 'foo']), pa.array(['foo', None, 'bar'])), (pa.array(['foo', None, 'bar', 'foo'], pa.large_binary()), pa.array(['foo', None, 'bar'], pa.large_binary()))]
    for arr, expected in cases:
        result = arr.unique()
        assert result.equals(expected)
        result = pa.chunked_array([arr]).unique()
        assert result.equals(expected)