from __future__ import print_function, absolute_import, division, unicode_literals
import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def Xtest_round_trip_non_exp_trailing_dot(self):
    data = round_trip('        ')
    print(data)