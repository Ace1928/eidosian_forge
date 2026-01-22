import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
class TestEmptyValueBeforeComments:

    def test_issue_25a(self):
        round_trip('        - a: b\n          c: d\n          d:  # foo\n          - e: f\n        ')

    def test_issue_25a1(self):
        round_trip('        - a: b\n          c: d\n          d:  # foo\n            e: f\n        ')

    def test_issue_25b(self):
        round_trip('        var1: #empty\n        var2: something #notempty\n        ')

    def test_issue_25c(self):
        round_trip('        params:\n          a: 1 # comment a\n          b:   # comment b\n          c: 3 # comment c\n        ')

    def test_issue_25c1(self):
        round_trip('        params:\n          a: 1 # comment a\n          b:   # comment b\n          # extra\n          c: 3 # comment c\n        ')

    def test_issue_25_00(self):
        round_trip('        params:\n          a: 1 # comment a\n          b:   # comment b\n        ')

    def test_issue_25_01(self):
        round_trip('        a:        # comment 1\n                  #  comment 2\n        - b:      #   comment 3\n            c: 1  #    comment 4\n        ')

    def test_issue_25_02(self):
        round_trip('        a:        # comment 1\n                  #  comment 2\n        - b: 2    #   comment 3\n        ')

    def test_issue_25_03(self):
        s = '        a:        # comment 1\n                  #  comment 2\n          - b: 2  #   comment 3\n        '
        round_trip(s, indent=4, block_seq_indent=2)

    def test_issue_25_04(self):
        round_trip('        a:        # comment 1\n                  #  comment 2\n          b: 1    #   comment 3\n        ')

    def test_flow_seq_within_seq(self):
        round_trip('        # comment 1\n        - a\n        - b\n        # comment 2\n        - c\n        - d\n        # comment 3\n        - [e]\n        - f\n        # comment 4\n        - []\n        ')