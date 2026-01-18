import re
import unittest
from wsme import exc
from wsme import types
def test_wsattr_del(self):

    class MyType(object):
        a = types.wsattr(int)
    types.register_type(MyType)
    value = MyType()
    value.a = 5
    assert value.a == 5
    del value.a
    assert value.a is types.Unset