import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_settext(self):
    assert self.call('argtypes/settext', value='ã\x81®', _rt=wsme.types.text) == 'ã\x81®'