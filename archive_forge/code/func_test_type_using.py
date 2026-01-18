import logging
import re
import warnings
from passlib import hash
from passlib.utils.compat import unicode
from passlib.tests.utils import HandlerCase, TEST_MODE
from passlib.tests.test_handlers import UPASS_TABLE, PASS_TABLE_UTF8
def test_type_using(self):
    handler = self.handler
    orig_type = handler.type
    for alt_type in handler.type_values:
        if alt_type != orig_type:
            break
    else:
        raise AssertionError('expected to find alternate type: default=%r values=%r' % (orig_type, handler.type_values))

    def effective_type(cls):
        return cls(use_defaults=True).type
    subcls = handler.using()
    self.assertEqual(subcls.type, orig_type)
    subcls = handler.using(type=alt_type)
    self.assertEqual(subcls.type, alt_type)
    self.assertEqual(handler.type, orig_type)
    self.assertEqual(effective_type(subcls), alt_type)
    self.assertEqual(effective_type(handler), orig_type)
    self.assertRaises(ValueError, handler.using, type='xXx')
    subcls = handler.using(type=alt_type)
    self.assertEqual(subcls.type, alt_type)
    self.assertEqual(handler.type, orig_type)
    self.assertEqual(effective_type(handler.using(type='I')), 'i')