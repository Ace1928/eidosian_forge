from collections import OrderedDict
from decimal import Decimal
import io
import itertools
import json
import string
import unittest
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.json import read_json, ReadOptions, ParseOptions
def test_small_random_json(self):
    data, expected = make_random_json(num_cols=2, num_rows=10)
    table = self.read_bytes(data)
    assert table.schema == expected.schema
    assert table.equals(expected)
    assert table.to_pydict() == expected.to_pydict()