import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_setnesteddict(self):
    value = {b'o1': {'inner': {'aint': 54}}, b'o2': {'inner': {'aint': 55}}}
    r = self.call('argtypes/setnesteddict', value=(value, {bytes: NestedOuter}), _rt={bytes: NestedOuter})
    print(r)
    self.assertEqual(r, value)