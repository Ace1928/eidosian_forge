from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
def validate_add_parameters(self, device_id=None, external_acceleration_type=None, storage_pool_id=None, storage_pool_name=None, acceleration_pool_id=None, acceleration_pool_name=None):
    """Validate the add device parameters"""
    if device_id:
        error_msg = 'Addition of device is allowed using device_name only, device_id given.'
        LOG.info(error_msg)
        self.module.fail_json(msg=error_msg)
    if external_acceleration_type and storage_pool_id is None and (storage_pool_name is None) and (acceleration_pool_id is None) and (acceleration_pool_name is None):
        error_msg = 'Storage Pool ID/name or Acceleration Pool ID/name is mandatory along with external_acceleration_type.'
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)