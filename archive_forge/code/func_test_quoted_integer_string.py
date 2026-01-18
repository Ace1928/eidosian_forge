from __future__ import print_function
import pytest
import platform
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_quoted_integer_string(self):
    round_trip("\n        a: '12345'\n        ")