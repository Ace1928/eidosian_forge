from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.dimensiondata import DimensionDataModule, UnknownNetworkError
def _delete_vlan(self, vlan):
    try:
        self.driver.ex_delete_vlan(vlan)
        if self.wait:
            self._wait_for_vlan_state(vlan, 'NOT_FOUND')
    except DimensionDataAPIException as api_exception:
        self.module.fail_json(msg='Failed to delete VLAN "{0}" due to unexpected error from the CloudControl API: {1}'.format(vlan.id, api_exception.msg))