from __future__ import print_function
import pytest  # NOQA
from .roundtrip import YAML  # does an automatic dedent on load
def test_root_literal_doc_indent_marker(self):
    yaml = YAML()
    yaml.explicit_start = True
    inp = '\n        --- |2\n           some more\n          text\n        '
    d = yaml.load(inp)
    print(type(d), repr(d))
    yaml.round_trip(inp)