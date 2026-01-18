from unittest import mock
import testtools
from testtools import matchers
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import driver
def test_vendor_passthru_methods(self):
    vendor_methods = self.mgr.get_vendor_passthru_methods(DRIVER1['name'])
    expect = [('GET', '/v1/drivers/%s/vendor_passthru/methods' % DRIVER1['name'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(DRIVER_VENDOR_PASSTHRU_METHOD, vendor_methods)