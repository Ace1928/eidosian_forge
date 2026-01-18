from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.dns import dnsutils
@mock.patch.object(dnsutils.DNSUtils, '_get_zone')
def test_zone_update_from_ds(self, mock_get_zone):
    mock_zone = mock.MagicMock(DsIntegrated=True, ZoneType=constants.DNS_ZONE_TYPE_PRIMARY)
    mock_get_zone.return_value = mock_zone
    self._dnsutils.zone_update(mock.sentinel.zone_name)
    mock_get_zone.assert_called_once_with(mock.sentinel.zone_name, ignore_missing=False)
    mock_zone.UpdateFromDS.assert_called_once_with()