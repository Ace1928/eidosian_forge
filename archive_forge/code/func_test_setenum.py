import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_setenum(self):
    value = b'v1'
    r = self.call('argtypes/setenum', value=value, _rt=myenumtype)
    self.assertEqual(r, value)