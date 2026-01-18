import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_pos_sequence(self):
    data = load('\n        - a\n        - b\n        - c\n        # next one!\n        - klm\n        - d\n        ')
    assert data.lc.item(3) == (4, 2)