import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
class RestOnlyProtocolTestCase(ProtocolTestCase):

    def test_body_list(self):
        r = self.call('bodytypes/setlist', body=([10], [int]), _rt=int)
        self.assertEqual(r, 10)

    def test_body_dict(self):
        r = self.call('bodytypes/setdict', body=({'test': 10}, {wsme.types.text: int}), _rt=int)
        self.assertEqual(r, 10)