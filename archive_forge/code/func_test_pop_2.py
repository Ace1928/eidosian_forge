import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_pop_2(self):
    d = round_trip_load(self.ins)
    d['ab'].pop(2)
    y = round_trip_dump(d, indent=2)
    print(y)
    assert y == dedent('        ab:\n        - a      # a\n        - b      # b\n        - d      # d\n\n        de:\n        - 1\n        - 2\n        ')