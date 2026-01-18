import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_return_namedattrsobj(self):
    r = self.call('returntypes/getnamedattrsobj', _rt=NamedAttrsObject)
    self.assertEqual(r, {'attr.1': 5, 'attr.2': 6})