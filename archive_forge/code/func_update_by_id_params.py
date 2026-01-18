from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.meraki.plugins.plugin_utils.meraki import (
from ansible_collections.cisco.meraki.plugins.plugin_utils.exceptions import (
def update_by_id_params(self):
    new_object_params = {}
    if self.new_object.get('enabled') is not None or self.new_object.get('enabled') is not None:
        new_object_params['enabled'] = self.new_object.get('enabled')
    if self.new_object.get('dropUntaggedTraffic') is not None or self.new_object.get('drop_untagged_traffic') is not None:
        new_object_params['dropUntaggedTraffic'] = self.new_object.get('dropUntaggedTraffic')
    if self.new_object.get('type') is not None or self.new_object.get('type') is not None:
        new_object_params['type'] = self.new_object.get('type') or self.new_object.get('type')
    if self.new_object.get('vlan') is not None or self.new_object.get('vlan') is not None:
        new_object_params['vlan'] = self.new_object.get('vlan') or self.new_object.get('vlan')
    if self.new_object.get('allowedVlans') is not None or self.new_object.get('allowed_vlans') is not None:
        new_object_params['allowedVlans'] = self.new_object.get('allowedVlans') or self.new_object.get('allowed_vlans')
    if self.new_object.get('accessPolicy') is not None or self.new_object.get('access_policy') is not None:
        new_object_params['accessPolicy'] = self.new_object.get('accessPolicy') or self.new_object.get('access_policy')
    if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
        new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
    if self.new_object.get('portId') is not None or self.new_object.get('port_id') is not None:
        new_object_params['portId'] = self.new_object.get('portId') or self.new_object.get('port_id')
    return new_object_params