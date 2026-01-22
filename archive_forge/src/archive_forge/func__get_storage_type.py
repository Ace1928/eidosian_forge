from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _get_storage_type(self):
    for sd_type in ['nfs', 'iscsi', 'posixfs', 'glusterfs', 'fcp', 'localfs', 'managed_block_storage']:
        if self.param(sd_type) is not None:
            return sd_type