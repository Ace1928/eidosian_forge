from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import pytest  # NOQA
from .roundtrip import round_trip, round_trip_load, round_trip_dump, dedent, YAML
def test_roundtrip_inline_list(self):
    s = 'a: [a, b, c]\n'
    output = rt(s)
    assert s == output