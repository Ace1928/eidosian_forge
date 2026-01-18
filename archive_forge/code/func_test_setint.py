import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_setint(self):
    r = self.call('argtypes/setint', value=3, _rt=int)
    self.assertEqual(r, 3)