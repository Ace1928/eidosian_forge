from unittest import mock
from os_brick import exception
from os_brick.initiator.connectors import fibre_channel_s390x
from os_brick.initiator import linuxfc
from os_brick.tests.initiator import test_connector
@mock.patch.object(fibre_channel_s390x.FibreChannelConnectorS390X, '_get_possible_devices', return_value=[('', 3, 5, 2)])
@mock.patch.object(linuxfc.LinuxFibreChannelS390X, 'get_fc_hbas_info', return_value=[])
@mock.patch.object(linuxfc.LinuxFibreChannelS390X, 'deconfigure_scsi_device')
def test_remove_devices(self, mock_deconfigure_scsi_device, mock_get_fc_hbas_info, mock_get_possible_devices):
    exc = exception.ExceptionChainer()
    connection_properties = {'targets': [5, 2]}
    self.connector._remove_devices(connection_properties, devices=None, device_info=None, force=False, exc=exc)
    mock_deconfigure_scsi_device.assert_called_with(3, 5, '0x0002000000000000')
    mock_get_fc_hbas_info.assert_called_once_with()
    mock_get_possible_devices.assert_called_once_with([], [5, 2], None)
    self.assertFalse(bool(exc))