import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_return_nestedarray(self):
    r = self.call('returntypes/getnestedarray', _rt=[NestedOuter])
    self.assertEqual(r, [{'inner': {'aint': 0}}, {'inner': {'aint': 0}}])