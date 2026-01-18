from __future__ import print_function
import pytest  # NOQA
from .roundtrip import YAML  # does an automatic dedent on load
def test_root_folding_scalar_no_indent_special(self):
    yaml = YAML()
    s = '%!PS-Adobe-2.0'
    inp = '\n        --- >\n        {}\n        '
    d = yaml.load(inp.format(s))
    print(d)
    assert d == s + '\n'