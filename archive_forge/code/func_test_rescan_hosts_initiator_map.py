import os.path
from unittest import mock
import ddt
from os_brick.initiator import linuxfc
from os_brick.tests import base
def test_rescan_hosts_initiator_map(self):
    """Test FC rescan with initiator map and not every HBA connected."""
    get_chan_results = [([['2', '3', 1], ['4', '5', 1]], set()), ([['6', '7', 1]], set())]
    hbas, con_props = self.__get_rescan_info(zone_manager=True)
    hbas.append({'device_path': '/sys/devices/pci0000:00/0000:00:02.0/0000:04:00.2/host8/fc_host/host8', 'host_device': 'host8', 'node_name': '50014380186af83g', 'port_name': '50014380186af83h'})
    with mock.patch.object(self.lfc, '_execute', return_value=None) as execute_mock, mock.patch.object(self.lfc, '_get_hba_channel_scsi_target_lun', side_effect=get_chan_results) as mock_get_chan:
        self.lfc.rescan_hosts(hbas, con_props)
        expected_commands = [mock.call('tee', '-a', '/sys/class/scsi_host/host6/scan', process_input='2 3 1', root_helper=None, run_as_root=True), mock.call('tee', '-a', '/sys/class/scsi_host/host6/scan', process_input='4 5 1', root_helper=None, run_as_root=True), mock.call('tee', '-a', '/sys/class/scsi_host/host7/scan', process_input='6 7 1', root_helper=None, run_as_root=True)]
        execute_mock.assert_has_calls(expected_commands)
        self.assertEqual(len(expected_commands), execute_mock.call_count)
        expected_calls = [mock.call(hbas[0], con_props), mock.call(hbas[1], con_props)]
        mock_get_chan.assert_has_calls(expected_calls)