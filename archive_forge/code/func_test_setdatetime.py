import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_setdatetime(self):
    value = datetime.datetime(2008, 4, 6, 12, 12, 15)
    r = self.call('argtypes/setdatetime', value=value, _rt=datetime.datetime)
    self.assertEqual(r, datetime.datetime(2008, 4, 6, 12, 12, 15))