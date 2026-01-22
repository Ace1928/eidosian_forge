from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
class SnapshotPolicyCreateHandler:

    def handle(self, con_object, con_params, snapshot_policy_details, access_mode, auto_snapshot_creation_cadence_in_min):
        if con_params['state'] == 'present' and (not snapshot_policy_details):
            if con_params['snapshot_policy_id']:
                con_object.module.fail_json(msg='Creation of snapshot policy is allowed using snapshot_policy_name only, snapshot_policy_id given.')
            snap_pol_id = con_object.create_snapshot_policy(snapshot_policy_name=con_params['snapshot_policy_name'], access_mode=access_mode, secure_snapshots=con_params['secure_snapshots'], auto_snapshot_creation_cadence_in_min=auto_snapshot_creation_cadence_in_min, num_of_retained_snapshots_per_level=con_params['num_of_retained_snapshots_per_level'])
            con_object.result['changed'] = True
            if snap_pol_id:
                snapshot_policy_details = con_object.get_snapshot_policy(snap_pol_name=con_params['snapshot_policy_name'], snap_pol_id=con_params['snapshot_policy_id'])
                msg = f'snapshot policy created successfully, fetched snapshot_policy details {str(snapshot_policy_details)}'
                LOG.info(msg)
        SnapshotPolicyModifyHandler().handle(con_object, con_params, snapshot_policy_details, auto_snapshot_creation_cadence_in_min)