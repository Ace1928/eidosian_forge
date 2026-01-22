from __future__ import print_function
import pytest  # NOQA
import json
class TestJSONNumbers:

    def test_json_number_float(self):
        for x in (y.split('#')[0].strip() for y in '\n        1.0  # should fail on YAML spec on 1-9 allowed as single digit\n        -1.0\n        1e-06\n        3.1e-5\n        3.1e+5\n        3.1e5  # should fail on YAML spec: no +- after e\n        '.splitlines()):
            if not x:
                continue
            res = load(x)
            assert isinstance(res, float)

    def test_json_number_int(self):
        for x in (y.split('#')[0].strip() for y in '\n        42\n        '.splitlines()):
            if not x:
                continue
            res = load(x, int)
            assert isinstance(res, int)