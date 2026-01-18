from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.powerflex_base \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import copy
def validate_rmcache_size_parameter(self, rmcache_enabled, rmcache_size):
    """Validate the input parameters"""
    if rmcache_size is not None and rmcache_enabled is False:
        error_msg = 'RM cache size can be set only when RM cache is enabled, please enable it along with RM cache size.'
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)