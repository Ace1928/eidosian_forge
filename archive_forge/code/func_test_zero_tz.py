import copy
import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_zero_tz(self):
    round_trip('\n        - 2011-10-02T15:45:00+0\n        ')