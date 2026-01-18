import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_setbinary(self):
    value = binarysample
    r = self.call('argtypes/setbinary', value=(value, wsme.types.binary), _rt=wsme.types.binary) == value
    print(r)