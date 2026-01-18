import pytest
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
@pytest.mark.xfail(strict=True)
def test_set_comment_before_tag(self):
    round_trip('\n        # the beginning\n        !!set\n        # or this one?\n        ? a\n        # next one is B (lowercase)\n        ? b  #  You see? Promised you.\n        ? c\n        # this is the end\n        ')