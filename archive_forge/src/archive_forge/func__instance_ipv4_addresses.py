from __future__ import absolute_import, division, print_function
import copy
import datetime
import os
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.lxd import LXDClient, LXDClientException
from ansible.module_utils.six.moves.urllib.parse import urlencode
def _instance_ipv4_addresses(self, ignore_devices=None):
    ignore_devices = ['lo'] if ignore_devices is None else ignore_devices
    data = (self._get_instance_state_json() or {}).get('metadata', None) or {}
    network = dict(((k, v) for k, v in (data.get('network', None) or {}).items() if k not in ignore_devices))
    addresses = dict(((k, [a['address'] for a in v['addresses'] if a['family'] == 'inet']) for k, v in network.items()))
    return addresses