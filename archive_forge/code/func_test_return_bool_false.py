import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_return_bool_false(self):
    r = self.call('returntypes/getboolfalse', _rt=bool)
    assert not r