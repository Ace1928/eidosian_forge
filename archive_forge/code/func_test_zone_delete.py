from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.dns import dnsutils
@mock.patch.object(dnsutils.DNSUtils, '_get_zone')
def test_zone_delete(self, mock_get_zone):
    self._dnsutils.zone_delete(mock.sentinel.zone_name)
    mock_get_zone.assert_called_once_with(mock.sentinel.zone_name)
    mock_get_zone.return_value.Delete_.assert_called_once_with()