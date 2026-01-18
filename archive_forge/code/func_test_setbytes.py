import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_setbytes(self):
    assert self.call('argtypes/setbytes', value=b'astring', _rt=wsme.types.bytes) == b'astring'