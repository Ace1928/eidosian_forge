from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import pytest  # NOQA
from .roundtrip import round_trip, round_trip_load, round_trip_dump, dedent, YAML
def test_roundtrip_mapping_of_inline_lists_comments(self):
    s = dedent('        # comment A\n        a: [a, b, c]\n        # comment B\n        j: [k, l, m]\n        ')
    output = rt(s)
    assert s == output