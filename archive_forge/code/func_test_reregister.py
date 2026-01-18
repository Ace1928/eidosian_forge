import re
import unittest
from wsme import exc
from wsme import types
def test_reregister(self):

    class TempType(object):
        pass
    types.registry.register(TempType)
    v = types.registry.lookup('TempType')
    self.assertIs(v, TempType)
    types.registry.reregister(TempType)
    after = types.registry.lookup('TempType')
    self.assertIs(after, TempType)