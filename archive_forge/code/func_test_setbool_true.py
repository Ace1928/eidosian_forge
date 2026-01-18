import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_setbool_true(self):
    r = self.call('argtypes/setbool', value=True, _rt=bool)
    assert r