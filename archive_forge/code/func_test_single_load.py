from __future__ import print_function
import sys
import pytest
def test_single_load(self):
    d = get_yaml().load(single_doc)
    print(d)
    print(type(d[0]))
    assert d == single_data