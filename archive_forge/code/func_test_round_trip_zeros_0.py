from __future__ import print_function, absolute_import, division, unicode_literals
import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_round_trip_zeros_0(self):
    data = round_trip('        - 0.\n        - +0.\n        - -0.\n        - 0.0\n        - +0.0\n        - -0.0\n        - 0.00\n        - +0.00\n        - -0.00\n        ')
    print(data)
    for d in data:
        assert -1e-05 < d < 1e-05