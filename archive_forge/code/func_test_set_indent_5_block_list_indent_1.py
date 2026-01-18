from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import pytest  # NOQA
from .roundtrip import round_trip, round_trip_load, round_trip_dump, dedent, YAML
def test_set_indent_5_block_list_indent_1(self):
    inp = '\n        a:\n         -   b: c\n         -   1\n         -   d:\n              -   2\n        '
    round_trip(inp, indent=5, block_seq_indent=1)