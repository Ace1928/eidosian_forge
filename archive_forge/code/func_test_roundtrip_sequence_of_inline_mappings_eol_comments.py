from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import pytest  # NOQA
from .roundtrip import round_trip, round_trip_load, round_trip_dump, dedent, YAML
def test_roundtrip_sequence_of_inline_mappings_eol_comments(self):
    s = dedent('        # comment A\n        - {a: 1, b: hallo}  # comment B\n        - {j: fka, k: 42}  # comment C\n        ')
    output = rt(s)
    assert s == output