from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import re
def validate_subset(self, api_version, subset):
    if float(api_version) < MIN_SUPPORTED_POWERFLEX_MANAGER_VERSION and subset and set(subset).issubset(POWERFLEX_MANAGER_GATHER_SUBSET):
        self.module.exit_json(msg=UNSUPPORTED_SUBSET_FOR_VERSION, skipped=True)