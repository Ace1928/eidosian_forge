import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def assertDateTimeEquals(self, a, b):
    self.assertTypedEquals(a, b, wsme.utils.parse_isodatetime)