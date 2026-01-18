import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_return_objectdictattribute(self):
    r = self.call('returntypes/getobjectdictattribute', _rt=NestedOuter)
    self.assertEqual(r, {'inner': {'aint': 0}, 'inner_dict': {'12': {'aint': 12}, '13': {'aint': 13}}})