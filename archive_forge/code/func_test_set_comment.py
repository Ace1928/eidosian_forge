import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_set_comment(self):
    round_trip('\n        !!set\n        # the beginning\n        ? a\n        # next one is B (lowercase)\n        ? b  #  You see? Promised you.\n        ? c\n        # this is the end\n        ')