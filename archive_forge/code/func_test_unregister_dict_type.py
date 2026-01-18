import re
import unittest
from wsme import exc
from wsme import types
def test_unregister_dict_type(self):

    class TempType(object):
        pass
    t = {str: TempType}
    types.registry.register(t)
    self.assertNotEqual(types.registry.dict_types, set())
    types.registry._unregister(t)
    self.assertEqual(types.registry.dict_types, set())