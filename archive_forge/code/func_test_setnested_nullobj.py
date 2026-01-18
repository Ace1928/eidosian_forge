import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_setnested_nullobj(self):
    value = {'inner': None}
    r = self.call('argtypes/setnested', value=(value, NestedOuter), _rt=NestedOuter)
    self.assertEqual(r, value)