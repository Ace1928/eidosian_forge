from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import ConnectionError
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def validate_feature(module, mode='show'):
    """Some features may need to be mapped due to inconsistency
    between how they appear from "show feature" output and
    how they are configured"""
    feature = module.params['feature']
    try:
        info = get_capabilities(module)
        device_info = info.get('device_info', {})
        os_version = device_info.get('network_os_version', '')
        os_platform = device_info.get('network_os_platform', '')
    except ConnectionError:
        os_version = ''
        os_platform = ''
    if '8.1' in os_version:
        feature_to_be_mapped = {'show': {'nv overlay': 'nve', 'vn-segment-vlan-based': 'vnseg_vlan', 'hsrp': 'hsrp_engine', 'fabric multicast': 'fabric_mcast', 'scp-server': 'scpServer', 'sftp-server': 'sftpServer', 'sla responder': 'sla_responder', 'sla sender': 'sla_sender', 'ssh': 'sshServer', 'tacacs+': 'tacacs', 'telnet': 'telnetServer', 'ethernet-link-oam': 'elo'}, 'config': {'nve': 'nv overlay', 'vnseg_vlan': 'vn-segment-vlan-based', 'hsrp_engine': 'hsrp', 'fabric_mcast': 'fabric multicast', 'scpServer': 'scp-server', 'sftpServer': 'sftp-server', 'sla_sender': 'sla sender', 'sla_responder': 'sla responder', 'sshServer': 'ssh', 'tacacs': 'tacacs+', 'telnetServer': 'telnet', 'elo': 'ethernet-link-oam'}}
    else:
        feature_to_be_mapped = {'show': {'nv overlay': 'nve', 'vn-segment-vlan-based': 'vnseg_vlan', 'hsrp': 'hsrp_engine', 'fabric multicast': 'fabric_mcast', 'scp-server': 'scpServer', 'sftp-server': 'sftpServer', 'sla responder': 'sla_responder', 'sla sender': 'sla_sender', 'ssh': 'sshServer', 'tacacs+': 'tacacs', 'telnet': 'telnetServer', 'ethernet-link-oam': 'elo', 'port-security': 'eth_port_sec'}, 'config': {'nve': 'nv overlay', 'vnseg_vlan': 'vn-segment-vlan-based', 'hsrp_engine': 'hsrp', 'fabric_mcast': 'fabric multicast', 'scpServer': 'scp-server', 'sftpServer': 'sftp-server', 'sla_sender': 'sla sender', 'sla_responder': 'sla responder', 'sshServer': 'ssh', 'tacacs': 'tacacs+', 'telnetServer': 'telnet', 'elo': 'ethernet-link-oam', 'eth_port_sec': 'port-security'}}
    if os_platform.startswith('DS-'):
        feature_to_be_mapped = get_mds_mapping_features()
    if feature in feature_to_be_mapped[mode]:
        feature = feature_to_be_mapped[mode][feature]
    return feature