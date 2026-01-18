import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_return_text(self):
    r = self.call('returntypes/gettext', _rt=wsme.types.text)
    self.assertEqual(r, 'ã\x81®')