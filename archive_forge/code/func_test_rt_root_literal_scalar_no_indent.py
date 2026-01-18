from __future__ import print_function
import pytest  # NOQA
from .roundtrip import YAML  # does an automatic dedent on load
def test_rt_root_literal_scalar_no_indent(self):
    yaml = YAML()
    yaml.explicit_start = True
    s = 'testing123'
    ys = '\n        --- |\n        {}\n        '
    ys = ys.format(s)
    d = yaml.load(ys)
    yaml.dump(d, compare=ys)