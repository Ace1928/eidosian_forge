from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
from datetime import datetime
def to_update(self, fs_snapshot, description=None, auto_del=None, expiry_time=None, fs_access_type=None):
    """Determines whether to update the snapshot or not"""
    snap_modify_dict = dict()
    if fs_access_type and fs_access_type != fs_snapshot.access_type:
        error_message = 'Modification of access type is not allowed.'
        LOG.error(error_message)
        self.module.fail_json(msg=error_message)
    if expiry_time and fs_snapshot.is_auto_delete and (auto_del is None or auto_del):
        self.module.fail_json(msg='expiry_time can be assigned when auto delete is False.')
    if auto_del is not None:
        if fs_snapshot.expiration_time:
            error_msg = 'expiry_time for filesystem snapshot is set. Once it is set then snapshot cannot be assigned to auto_delete policy.'
            self.module.fail_json(msg=error_msg)
        if auto_del != fs_snapshot.is_auto_delete:
            snap_modify_dict['is_auto_delete'] = auto_del
    if description is not None and description != fs_snapshot.description:
        snap_modify_dict['description'] = description
    if to_update_expiry_time(fs_snapshot, expiry_time):
        snap_modify_dict['expiry_time'] = expiry_time
    LOG.info('Snapshot modification details: %s', snap_modify_dict)
    return snap_modify_dict