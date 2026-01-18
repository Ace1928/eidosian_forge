from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def volume_modify_required(self, obj_vol, cap_unit):
    """Check if volume modification is required
            :param obj_vol: volume instance
            :param cap_unit: capacity unit
            :return: Boolean value to indicate if modification is required
        """
    try:
        to_update = {}
        new_vol_name = self.module.params['new_vol_name']
        if new_vol_name and obj_vol.name != new_vol_name:
            to_update.update({'name': new_vol_name})
        description = self.module.params['description']
        if description and obj_vol.description != description:
            to_update.update({'description': description})
        size = self.module.params['size']
        if size and cap_unit:
            size_byte = int(utils.get_size_bytes(size, cap_unit))
            if size_byte < obj_vol.size_total:
                self.module.fail_json(msg='Volume size can be expanded only')
            elif size_byte > obj_vol.size_total:
                to_update.update({'size': size_byte})
        compression = self.module.params['compression']
        if compression is not None and compression != obj_vol.is_data_reduction_enabled:
            to_update.update({'is_compression': compression})
        advanced_dedup = self.module.params['advanced_dedup']
        if advanced_dedup is not None and advanced_dedup != obj_vol.is_advanced_dedup_enabled:
            to_update.update({'is_advanced_dedup_enabled': advanced_dedup})
        is_thin = self.module.params['is_thin']
        if is_thin is not None and is_thin != obj_vol.is_thin_enabled:
            self.module.fail_json(msg='Modifying is_thin is not allowed')
        sp = self.module.params['sp']
        if sp and self.get_node_enum(sp) != obj_vol.current_node:
            to_update.update({'sp': self.get_node_enum(sp)})
        tiering_policy = self.module.params['tiering_policy']
        if tiering_policy and self.get_tiering_policy_enum(tiering_policy) != obj_vol.tiering_policy:
            to_update.update({'tiering_policy': self.get_tiering_policy_enum(tiering_policy)})
        if self.param_io_limit_pol_id:
            if not obj_vol.io_limit_policy or self.param_io_limit_pol_id != obj_vol.io_limit_policy.id:
                to_update.update({'io_limit_policy': self.param_io_limit_pol_id})
        if self.param_snap_schedule_name:
            if not obj_vol.snap_schedule or self.param_snap_schedule_name != obj_vol.snap_schedule.name:
                to_update.update({'snap_schedule': self.param_snap_schedule_name})
        if self.param_snap_schedule_name == '':
            if obj_vol.snap_schedule:
                to_update.update({'is_snap_schedule_paused': False})
            else:
                LOG.warn('No snapshot schedule is associated')
        LOG.debug('Volume to modify  Dict : %s', to_update)
        if len(to_update) > 0:
            return to_update
        else:
            return None
    except Exception as e:
        errormsg = 'Failed to determine if volume {0},requires modification, with error {1}'.format(obj_vol.name, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)