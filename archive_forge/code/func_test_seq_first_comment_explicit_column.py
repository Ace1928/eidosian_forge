from __future__ import print_function
import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_seq_first_comment_explicit_column(self):
    data = load('\n        - a\n        - b\n        - c\n        ')
    data.yaml_add_eol_comment('comment 1', key=1, column=6)
    exp = '\n        - a\n        - b   # comment 1\n        - c\n        '
    compare(data, exp)