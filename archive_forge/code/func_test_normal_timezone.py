import copy
import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_normal_timezone(self):
    round_trip('\n        - 2011-10-02T11:45:00-5\n        - 2011-10-02 11:45:00-5\n        - 2011-10-02T11:45:00-05:00\n        - 2011-10-02 11:45:00-05:00\n        ')