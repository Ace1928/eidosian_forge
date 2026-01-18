from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.dns import dnsutils
@mock.patch.object(dnsutils.DNSUtils, '_get_zone')
def test_zone_exists_false(self, mock_get_zone):
    mock_get_zone.return_value = None
    zone_already_exists = self._dnsutils.zone_exists(mock.sentinel.zone_name)
    mock_get_zone.assert_called_once_with(mock.sentinel.zone_name)
    self.assertFalse(zone_already_exists)