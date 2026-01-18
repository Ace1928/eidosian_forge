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
def test_unique_value_counts_dictionary_type():
    indices = pa.array([3, 0, 0, 0, 1, 1, 3, 0, 1, 3, 0, 1])
    dictionary = pa.array(['foo', 'bar', 'baz', 'qux'])
    arr = pa.DictionaryArray.from_arrays(indices, dictionary)
    unique_result = arr.unique()
    expected = pa.DictionaryArray.from_arrays(indices.unique(), dictionary)
    assert unique_result.equals(expected)
    result = arr.value_counts()
    assert result.field('values').equals(unique_result)
    assert result.field('counts').equals(pa.array([3, 5, 4], type='int64'))
    arr = pa.DictionaryArray.from_arrays(pa.array([], type='int64'), dictionary)
    unique_result = arr.unique()
    expected = pa.DictionaryArray.from_arrays(pa.array([], type='int64'), pa.array([], type='utf8'))
    assert unique_result.equals(expected)
    result = arr.value_counts()
    assert result.field('values').equals(unique_result)
    assert result.field('counts').equals(pa.array([], type='int64'))