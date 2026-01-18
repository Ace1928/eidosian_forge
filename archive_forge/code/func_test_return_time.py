import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_return_time(self):
    r = self.call('returntypes/gettime')
    self.assertTimeEquals(r, datetime.time(12))