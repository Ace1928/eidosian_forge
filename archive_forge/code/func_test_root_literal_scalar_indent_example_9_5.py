from __future__ import print_function
import pytest  # NOQA
from .roundtrip import YAML  # does an automatic dedent on load
def test_root_literal_scalar_indent_example_9_5(self):
    yaml = YAML()
    s = '%!PS-Adobe-2.0'
    inp = '\n        --- |\n          {}\n        '
    d = yaml.load(inp.format(s))
    print(d)
    assert d == s + '\n'