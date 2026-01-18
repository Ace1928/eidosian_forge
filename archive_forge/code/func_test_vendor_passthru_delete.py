from unittest import mock
import testtools
from testtools import matchers
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import driver
@mock.patch.object(driver.DriverManager, 'delete', autospec=True)
def test_vendor_passthru_delete(self, delete_mock):
    kwargs = {'driver_name': 'driver_name', 'method': 'method', 'http_method': 'DELETE'}
    final_path = 'driver_name/vendor_passthru/method'
    self.mgr.vendor_passthru(**kwargs)
    delete_mock.assert_called_once_with(mock.ANY, final_path, os_ironic_api_version=None, global_request_id=None)