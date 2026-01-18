from collections import OrderedDict
from collections.abc import Iterator
from functools import partial
import datetime
import sys
import pytest
import hypothesis as h
import hypothesis.strategies as st
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.types as types
import pyarrow.tests.strategies as past
def test_types_hashable():
    many_types = get_many_types()
    in_dict = {}
    for i, type_ in enumerate(many_types):
        assert hash(type_) == hash(type_)
        in_dict[type_] = i
    assert len(in_dict) == len(many_types)
    for i, type_ in enumerate(many_types):
        assert in_dict[type_] == i