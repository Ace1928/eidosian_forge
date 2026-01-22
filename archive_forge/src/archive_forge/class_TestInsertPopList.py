import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
class TestInsertPopList:
    """list insertion is more complex than dict insertion, as you
    need to move the values to subsequent keys on insert"""

    @property
    def ins(self):
        return '        ab:\n        - a      # a\n        - b      # b\n        - c\n        - d      # d\n\n        de:\n        - 1\n        - 2\n        '

    def test_insert_0(self):
        d = round_trip_load(self.ins)
        d['ab'].insert(0, 'xyz')
        y = round_trip_dump(d, indent=2)
        assert y == dedent('        ab:\n        - xyz\n        - a      # a\n        - b      # b\n        - c\n        - d      # d\n\n        de:\n        - 1\n        - 2\n        ')

    def test_insert_1(self):
        d = round_trip_load(self.ins)
        d['ab'].insert(4, 'xyz')
        y = round_trip_dump(d, indent=2)
        assert y == dedent('        ab:\n        - a      # a\n        - b      # b\n        - c\n        - d      # d\n\n        - xyz\n        de:\n        - 1\n        - 2\n        ')

    def test_insert_2(self):
        d = round_trip_load(self.ins)
        d['ab'].insert(1, 'xyz')
        y = round_trip_dump(d, indent=2)
        assert y == dedent('        ab:\n        - a      # a\n        - xyz\n        - b      # b\n        - c\n        - d      # d\n\n        de:\n        - 1\n        - 2\n        ')

    def test_pop_0(self):
        d = round_trip_load(self.ins)
        d['ab'].pop(0)
        y = round_trip_dump(d, indent=2)
        print(y)
        assert y == dedent('        ab:\n        - b      # b\n        - c\n        - d      # d\n\n        de:\n        - 1\n        - 2\n        ')

    def test_pop_1(self):
        d = round_trip_load(self.ins)
        d['ab'].pop(1)
        y = round_trip_dump(d, indent=2)
        print(y)
        assert y == dedent('        ab:\n        - a      # a\n        - c\n        - d      # d\n\n        de:\n        - 1\n        - 2\n        ')

    def test_pop_2(self):
        d = round_trip_load(self.ins)
        d['ab'].pop(2)
        y = round_trip_dump(d, indent=2)
        print(y)
        assert y == dedent('        ab:\n        - a      # a\n        - b      # b\n        - d      # d\n\n        de:\n        - 1\n        - 2\n        ')

    def test_pop_3(self):
        d = round_trip_load(self.ins)
        d['ab'].pop(3)
        y = round_trip_dump(d, indent=2)
        print(y)
        assert y == dedent('        ab:\n        - a      # a\n        - b      # b\n        - c\n        de:\n        - 1\n        - 2\n        ')