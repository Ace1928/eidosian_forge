from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
class SnapshotPolicyPauseHandler:

    def handle(self, con_object, con_params, snapshot_policy_details):
        if con_params['state'] == 'present' and con_params['pause'] is not None:
            con_object.result['changed'] = con_object.pause_snapshot_policy(snap_pol_details=snapshot_policy_details)
            snapshot_policy_details = con_object.get_snapshot_policy(snap_pol_name=con_params['snapshot_policy_name'], snap_pol_id=con_params['snapshot_policy_id'])
        SnapshotPolicyDeleteHandler().handle(con_object, con_params, snapshot_policy_details)