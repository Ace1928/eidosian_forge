import os.path
from unittest import mock
import ddt
from os_brick.initiator import linuxfc
from os_brick.tests import base
def test_rescan_hosts_port_not_found(self):
    """Test when we don't find the target ports."""
    get_chan_results = [([], {1}), ([], {1})]
    hbas, con_props = self.__get_rescan_info(zone_manager=True)
    con_props.pop('initiator_target_map')
    con_props.pop('initiator_target_lun_map')
    with mock.patch.object(self.lfc, '_get_hba_channel_scsi_target_lun', side_effect=get_chan_results), mock.patch.object(self.lfc, '_execute', side_effect=None) as execute_mock:
        self.lfc.rescan_hosts(hbas, con_props)
        expected_commands = [mock.call('tee', '-a', '/sys/class/scsi_host/host6/scan', process_input='- - 1', root_helper=None, run_as_root=True), mock.call('tee', '-a', '/sys/class/scsi_host/host7/scan', process_input='- - 1', root_helper=None, run_as_root=True)]
        execute_mock.assert_has_calls(expected_commands)
        self.assertEqual(len(expected_commands), execute_mock.call_count)