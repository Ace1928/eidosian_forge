import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_settext_empty(self):
    assert self.call('argtypes/settext', value='', _rt=wsme.types.text) == ''