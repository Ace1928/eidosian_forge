from __future__ import (absolute_import, division, print_function)
import itertools
import re
from ansible.module_utils.common._collections_compat import MutableMapping
from ansible.errors import AnsibleError
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import string_types
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.utils.display import Display
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def _get_agent_network_interfaces(self, node, vmid, vmtype):
    result = []
    try:
        ifaces = self._get_json('%s/api2/json/nodes/%s/%s/%s/agent/network-get-interfaces' % (self.proxmox_url, node, vmtype, vmid))['result']
        if 'error' in ifaces:
            if 'class' in ifaces['error']:
                errorClass = ifaces['error']['class']
                if errorClass in ['Unsupported']:
                    self.display.v('Retrieving network interfaces from guest agents on windows with older qemu-guest-agents is not supported')
                elif errorClass in ['CommandDisabled']:
                    self.display.v('Retrieving network interfaces from guest agents has been disabled')
            return result
        for iface in ifaces:
            result.append({'name': iface['name'], 'mac-address': iface['hardware-address'] if 'hardware-address' in iface else '', 'ip-addresses': ['%s/%s' % (ip['ip-address'], ip['prefix']) for ip in iface['ip-addresses']] if 'ip-addresses' in iface else []})
    except requests.HTTPError:
        pass
    return result