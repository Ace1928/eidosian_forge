from __future__ import print_function
import pytest  # NOQA
from .roundtrip import YAML  # does an automatic dedent on load
def test_root_literal_multi_doc(self):
    yaml = YAML(typ='safe', pure=True)
    s1 = 'abc'
    s2 = 'klm'
    inp = '\n        --- |-\n        {}\n        --- |\n        {}\n        '
    for idx, d1 in enumerate(yaml.load_all(inp.format(s1, s2))):
        print('d1:', d1)
        assert ['abc', 'klm\n'][idx] == d1