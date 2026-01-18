from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def update_smb_share(self, smb_share_obj, is_encryption_enabled=None, is_continuous_availability_enabled=None, is_abe_enabled=None, is_branch_cache_enabled=None, umask=None, description=None, offline_availability=None):
    """
        The Details of the SMB share will be updated in the function.
        """
    try:
        smb_share_obj.modify(is_encryption_enabled=is_encryption_enabled, is_con_avail_enabled=is_continuous_availability_enabled, is_abe_enabled=is_abe_enabled, is_branch_cache_enabled=is_branch_cache_enabled, umask=umask, description=description, offline_availability=offline_availability)
    except Exception as e:
        error_msg = 'Failed to Update parameters of SMB share %s with error %s' % (smb_share_obj.name, str(e))
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)