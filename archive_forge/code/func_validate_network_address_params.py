from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
import ipaddress
def validate_network_address_params(self, network_address):
    if '.' in network_address and (not is_valid_ip(network_address)):
        err_msg = 'Please enter valid IPV4 address for network address'
        LOG.error(err_msg)
        self.module.fail_json(msg=err_msg)
    if len(network_address) < 1 or len(network_address) > 63:
        err_msg = "'network_address' should be in range of 1 to 63 characters."
        LOG.error(err_msg)
        self.module.fail_json(msg=err_msg)
    if utils.has_special_char(network_address) or ' ' in network_address:
        err_msg = 'Please enter valid IPV4 address or host name for network address'
        LOG.error(err_msg)
        self.module.fail_json(msg=err_msg)