import pytest
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
class TestTagFailures:

    @pytest.mark.xfail(strict=True)
    def test_standard_short_tag(self):
        round_trip('        !!map\n        name: Anthon\n        location: Germany\n        language: python\n        ')

    def test_standard_short_tag_no_fail(self):
        inp = '\n        !!map\n        name: Anthon\n        location: Germany\n        language: python\n        '
        exp = '\n        name: Anthon\n        location: Germany\n        language: python\n        '
        assert round_trip_dump(round_trip_load(inp)) == dedent(exp)