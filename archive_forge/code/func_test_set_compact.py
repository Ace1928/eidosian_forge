from __future__ import print_function
import sys
import pytest  # NOQA
import platform
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_set_compact(self):
    round_trip('\n        !!set\n        ? a\n        ? b\n        ? c\n        ')