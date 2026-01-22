from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.dimensiondata import HAS_LIBCLOUD, DimensionDataModule
from ansible.module_utils.common.text.converters import to_native
def _delete_network(self, network):
    try:
        if self.mcp_version == '1.0':
            deleted = self.driver.ex_delete_network(network)
        else:
            deleted = self.driver.ex_delete_network_domain(network)
        if deleted:
            self.module.exit_json(changed=True, msg='Deleted network with id %s' % network.id)
        self.module.fail_json('Unexpected failure deleting network with id %s' % network.id)
    except DimensionDataAPIException as e:
        self.module.fail_json(msg='Failed to delete network: %s' % to_native(e), exception=traceback.format_exc())