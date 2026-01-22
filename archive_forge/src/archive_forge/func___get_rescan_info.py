import os.path
from unittest import mock
import ddt
from os_brick.initiator import linuxfc
from os_brick.tests import base
@staticmethod
def __get_rescan_info(zone_manager=False):
    connection_properties = {'initiator_target_map': {'50014380186af83c': ['514f0c50023f6c00'], '50014380186af83e': ['514f0c50023f6c01']}, 'initiator_target_lun_map': {'50014380186af83c': [('514f0c50023f6c00', 1)], '50014380186af83e': [('514f0c50023f6c01', 1)]}, 'target_discovered': False, 'target_lun': 1, 'target_wwn': ['514f0c50023f6c00', '514f0c50023f6c01'], 'targets': [('514f0c50023f6c00', 1), ('514f0c50023f6c01', 1)]}
    hbas = [{'device_path': '/sys/devices/pci0000:00/0000:00:02.0/0000:04:00.0/host6/fc_host/host6', 'host_device': 'host6', 'node_name': '50014380186af83d', 'port_name': '50014380186af83c'}, {'device_path': '/sys/devices/pci0000:00/0000:00:02.0/0000:04:00.1/host7/fc_host/host7', 'host_device': 'host7', 'node_name': '50014380186af83f', 'port_name': '50014380186af83e'}]
    if not zone_manager:
        del connection_properties['initiator_target_map']
        del connection_properties['initiator_target_lun_map']
    return (hbas, connection_properties)