from __future__ import print_function
import sys
import pytest  # NOQA
import platform
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_set_out(self):
    import srsly.ruamel_yaml
    x = set(['a', 'b', 'c'])
    res = srsly.ruamel_yaml.dump(x, default_flow_style=False)
    assert res == dedent('\n        !!set\n        a: null\n        b: null\n        c: null\n        ')