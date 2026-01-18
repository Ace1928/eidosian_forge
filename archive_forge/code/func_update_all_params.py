from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.meraki.plugins.plugin_utils.meraki import (
from ansible_collections.cisco.meraki.plugins.plugin_utils.exceptions import (
def update_all_params(self):
    new_object_params = {}
    if self.new_object.get('destinations') is not None or self.new_object.get('destinations') is not None:
        new_object_params['destinations'] = self.new_object.get('destinations') or self.new_object.get('destinations')
    if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
        new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
    return new_object_params