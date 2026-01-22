from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def _tcp_udp_match(self, rule, protocol, start_port, end_port):
    return protocol in ['tcp', 'udp'] and protocol == rule['protocol'] and (start_port == int(rule['startport'])) and (end_port == int(rule['endport']))