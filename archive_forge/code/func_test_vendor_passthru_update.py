from unittest import mock
import testtools
from testtools import matchers
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import driver
@mock.patch.object(driver.DriverManager, 'update', autospec=True)
def test_vendor_passthru_update(self, update_mock):
    vendor_passthru_args = {'arg1': 'val1'}
    kwargs = {'driver_name': 'driver_name', 'method': 'method', 'args': vendor_passthru_args}
    final_path = 'driver_name/vendor_passthru/method'
    for http_method in ('POST', 'PUT', 'PATCH'):
        kwargs['http_method'] = http_method
        self.mgr.vendor_passthru(**kwargs)
        update_mock.assert_called_once_with(mock.ANY, final_path, vendor_passthru_args, http_method=http_method, os_ironic_api_version=None, global_request_id=None)
        update_mock.reset_mock()