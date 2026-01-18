import copy
import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_zero_fraction(self):
    inp = '\n        - 2011-10-02 16:45:00.0\n        '
    exp = '\n        - 2011-10-02 16:45:00\n        '
    round_trip(inp, exp)