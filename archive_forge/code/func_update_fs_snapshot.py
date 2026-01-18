from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@api_wrapper
def update_fs_snapshot(module, snapshot):
    """ Update/refresh fs snapshot. May also lock it. """
    refresh_changed = False
    lock_changed = False
    if not module.check_mode:
        if not module.params['snapshot_lock_only']:
            snap_is_locked = snapshot.get_lock_state() == 'LOCKED'
            if not snap_is_locked:
                if not module.check_mode:
                    snapshot.refresh_snapshot()
                refresh_changed = True
            else:
                msg = 'File system snapshot is locked and may not be refreshed'
                module.fail_json(msg=msg)
        check_snapshot_lock_options(module)
        lock_changed = manage_snapshot_locks(module, snapshot)
        if module.params['write_protected'] is not None:
            is_write_prot = snapshot.is_write_protected()
            desired_is_write_prot = module.params['write_protected']
            if is_write_prot != desired_is_write_prot:
                snapshot.update_field('write_protected', desired_is_write_prot)
    return refresh_changed or lock_changed