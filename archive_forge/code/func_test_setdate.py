import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_setdate(self):
    value = datetime.date(2008, 4, 6)
    r = self.call('argtypes/setdate', value=value, _rt=datetime.date)
    self.assertEqual(r, value)