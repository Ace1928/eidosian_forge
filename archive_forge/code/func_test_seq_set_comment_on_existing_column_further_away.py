from __future__ import print_function
import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_seq_set_comment_on_existing_column_further_away(self):
    """
        no comment line before or after, take the latest before
        the new position
        """
    data = load('\n        - a   # comment 1\n        - b\n        - c\n        - d\n        - e\n        - f     # comment 3\n        ')
    print(data._yaml_comment)
    data.yaml_add_eol_comment('comment 2', key=3)
    exp = '\n        - a   # comment 1\n        - b\n        - c\n        - d   # comment 2\n        - e\n        - f     # comment 3\n        '
    compare(data, exp)