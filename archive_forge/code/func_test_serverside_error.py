import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_serverside_error(self):
    try:
        res = self.call('witherrors/divide_by_zero')
        print(res)
        assert 'No error raised'
    except CallException as e:
        self.assertEqual(e.faultcode, 'Server')
        self.assertEqual(e.faultstring, zerodivisionerrormsg)
        assert e.debuginfo is not None