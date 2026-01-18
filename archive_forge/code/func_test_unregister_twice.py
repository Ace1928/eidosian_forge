import re
import unittest
from wsme import exc
from wsme import types
def test_unregister_twice(self):

    class TempType(object):
        pass
    types.registry.register(TempType)
    v = types.registry.lookup('TempType')
    self.assertIs(v, TempType)
    types.registry._unregister(TempType)
    types.registry._unregister(TempType)
    after = types.registry.lookup('TempType')
    self.assertIs(after, None)