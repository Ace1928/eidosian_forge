import re
import unittest
from wsme import exc
from wsme import types
def test_reregister_and_add_attr(self):

    class TempType(object):
        pass
    types.registry.register(TempType)
    attrs = types.list_attributes(TempType)
    self.assertEqual(attrs, [])
    TempType.one = str
    types.registry.reregister(TempType)
    after = types.list_attributes(TempType)
    self.assertNotEqual(after, [])