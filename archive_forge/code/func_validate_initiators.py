from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
import ipaddress
def validate_initiators(self, initiators):
    results = []
    for item in initiators:
        results.append(utils.is_initiator_valid(item))
    if False in results:
        error_message = 'One or more initiator provided is not valid, please provide valid initiators'
        LOG.error(error_message)
        self.module.fail_json(msg=error_message)