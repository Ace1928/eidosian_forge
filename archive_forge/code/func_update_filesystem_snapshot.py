from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
from datetime import datetime
def update_filesystem_snapshot(self, fs_snapshot, snap_modify_dict):
    try:
        duration = None
        if 'expiry_time' in snap_modify_dict and snap_modify_dict['expiry_time']:
            duration = convert_timestamp_to_sec(snap_modify_dict['expiry_time'], self.unity_conn.system_time)
        if duration and duration <= 0:
            self.module.fail_json(msg='expiry_time should be after the current system time.')
        if 'is_auto_delete' in snap_modify_dict and snap_modify_dict['is_auto_delete'] is not None:
            auto_delete = snap_modify_dict['is_auto_delete']
        else:
            auto_delete = None
        if 'description' in snap_modify_dict and (snap_modify_dict['description'] or len(snap_modify_dict['description']) == 0):
            description = snap_modify_dict['description']
        else:
            description = None
        fs_snapshot.modify(retentionDuration=duration, isAutoDelete=auto_delete, description=description)
        fs_snapshot.update()
    except Exception as e:
        error_msg = 'Failed to modify filesystem snapshot [name: %s , id: %s] with error %s.' % (fs_snapshot.name, fs_snapshot.id, str(e))
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)