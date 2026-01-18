from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.dns import dnsutils
@mock.patch.object(dnsutils.DNSUtils, '_get_zone')
def test_zone_modify(self, mock_get_zone):
    mock_zone = mock.MagicMock(AllowUpdate=mock.sentinel.allowupdate, DisableWINSRecordReplication=mock.sentinel.disablewins, Notify=mock.sentinel.notify, SecureSecondaries=mock.sentinel.securesecondaries)
    mock_get_zone.return_value = mock_zone
    self._dnsutils.zone_modify(mock.sentinel.zone_name, allow_update=None, disable_wins=mock.sentinel.disable_wins, notify=None, reverse=mock.sentinel.reverse, secure_secondaries=None)
    self.assertEqual(mock.sentinel.allowupdate, mock_zone.AllowUpdate)
    self.assertEqual(mock.sentinel.disable_wins, mock_zone.DisableWINSRecordReplication)
    self.assertEqual(mock.sentinel.notify, mock_zone.Notify)
    self.assertEqual(mock.sentinel.reverse, mock_zone.Reverse)
    self.assertEqual(mock.sentinel.securesecondaries, mock_zone.SecureSecondaries)
    mock_zone.put.assert_called_once_with()