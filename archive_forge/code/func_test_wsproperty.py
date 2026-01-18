import re
import unittest
from wsme import exc
from wsme import types
def test_wsproperty(self):

    class WithWSProp(object):

        def __init__(self):
            self._aint = 0

        def get_aint(self):
            return self._aint

        def set_aint(self, value):
            self._aint = value
        aint = types.wsproperty(int, get_aint, set_aint, mandatory=True)
    types.register_type(WithWSProp)
    print(WithWSProp._wsme_attributes)
    assert len(WithWSProp._wsme_attributes) == 1
    a = WithWSProp._wsme_attributes[0]
    assert a.key == 'aint'
    assert a.datatype == int
    assert a.mandatory
    o = WithWSProp()
    o.aint = 12
    assert o.aint == 12