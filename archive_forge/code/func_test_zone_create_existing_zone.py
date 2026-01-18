from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.dns import dnsutils
@mock.patch.object(dnsutils.DNSUtils, 'zone_exists')
def test_zone_create_existing_zone(self, mock_zone_exists):
    self.assertRaises(exceptions.DNSZoneAlreadyExists, self._dnsutils.zone_create, zone_name=mock.sentinel.zone_name, zone_type=mock.sentinel.zone_type, ds_integrated=mock.sentinel.ds_integrated)
    mock_zone_exists.assert_called_once_with(mock.sentinel.zone_name)