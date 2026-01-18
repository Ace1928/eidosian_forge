from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import pytest  # NOQA
from .roundtrip import round_trip, round_trip_load, round_trip_dump, dedent, YAML
def test_roundtrip_four_space_indents(self):
    s = 'a:\n-   foo\n-   bar\n'
    round_trip(s, indent=4)