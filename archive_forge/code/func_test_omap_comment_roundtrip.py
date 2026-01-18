import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_omap_comment_roundtrip(self):
    round_trip('\n        !!omap\n        - a: 1\n        - b: 2  # two\n        - c: 3  # three\n        - d: 4\n        ')