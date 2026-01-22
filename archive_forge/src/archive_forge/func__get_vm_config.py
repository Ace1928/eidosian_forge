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
def _get_vm_config(self, properties, node, vmid, vmtype, name):
    ret = self._get_json('%s/api2/json/nodes/%s/%s/%s/config' % (self.proxmox_url, node, vmtype, vmid))
    properties[self._fact('node')] = node
    properties[self._fact('vmid')] = vmid
    properties[self._fact('vmtype')] = vmtype
    plaintext_configs = ['description']
    for config in ret:
        key = self._fact(config)
        value = ret[config]
        try:
            if config == 'rootfs' or config.startswith(('virtio', 'sata', 'ide', 'scsi')):
                value = 'disk_image=' + value
            if config == 'tags':
                stripped_value = value.strip()
                if stripped_value:
                    parsed_key = key + '_parsed'
                    properties[parsed_key] = [tag.strip() for tag in stripped_value.replace(',', ';').split(';')]
            if config == 'agent':
                agent_enabled = 0
                try:
                    agent_enabled = int(value.split(',')[0])
                except ValueError:
                    if value.split(',')[0] == 'enabled=1':
                        agent_enabled = 1
                if agent_enabled:
                    agent_iface_value = self._get_agent_network_interfaces(node, vmid, vmtype)
                    if agent_iface_value:
                        agent_iface_key = self.to_safe('%s%s' % (key, '_interfaces'))
                        properties[agent_iface_key] = agent_iface_value
            if config == 'lxc':
                out_val = {}
                for k, v in value:
                    if k.startswith('lxc.'):
                        k = k[len('lxc.'):]
                    out_val[k] = v
                value = out_val
            if config not in plaintext_configs and isinstance(value, string_types) and all(('=' in v for v in value.split(','))):
                try:
                    value = dict((key.split('=', 1) for key in value.split(',')))
                except Exception:
                    continue
            properties[key] = value
        except NameError:
            return None