import logging
import re
import warnings
from passlib import hash
from passlib.utils.compat import unicode
from passlib.tests.utils import HandlerCase, TEST_MODE
from passlib.tests.test_handlers import UPASS_TABLE, PASS_TABLE_UTF8
def test_type_kwd(self):
    cls = self.handler
    self.assertTrue('type' in cls.setting_kwds)
    for value in cls.type_values:
        self.assertIsInstance(value, unicode)
    self.assertTrue('i' in cls.type_values)
    self.assertTrue('d' in cls.type_values)
    self.assertTrue(cls.type in cls.type_values)
    handler = cls
    hash = self.get_sample_hash()[1]
    kwds = handler.parsehash(hash)
    del kwds['type']
    handler(type=cls.type, **kwds)
    handler(**kwds)
    handler(use_defaults=True, **kwds)
    self.assertRaises(ValueError, handler, type='xXx', **kwds)