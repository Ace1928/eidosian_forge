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
def test_reconcile_across_blocks(self):
    first_row = b'{                               }\n'
    read_options = ReadOptions(block_size=len(first_row))
    for next_rows, expected_pylist in [(b'{"a": 0}', [None, 0]), (b'{"a": []}', [None, []]), (b'{"a": []}\n{"a": [[1]]}', [None, [], [[1]]]), (b'{"a": {}}', [None, {}]), (b'{"a": {}}\n{"a": {"b": {"c": 1}}}', [None, {'b': None}, {'b': {'c': 1}}])]:
        table = self.read_bytes(first_row + next_rows, read_options=read_options)
        expected = {'a': expected_pylist}
        assert table.to_pydict() == expected
        assert table.column('a').num_chunks > 1