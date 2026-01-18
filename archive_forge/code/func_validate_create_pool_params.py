from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def validate_create_pool_params(self, alert_threshold=None, pool_harvest_high_threshold=None, pool_harvest_low_threshold=None, snap_harvest_high_threshold=None, snap_harvest_low_threshold=None):
    """ Validates params for creating pool"""
    if alert_threshold and (alert_threshold < 50 or alert_threshold > 84):
        errormsg = 'Alert threshold is not in the allowed value range of 50 - 84'
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)
    if pool_harvest_high_threshold and (pool_harvest_high_threshold < 1 or pool_harvest_high_threshold > 99):
        errormsg = 'Pool harvest high threshold is not in the allowed value range of 1 - 99'
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)
    if pool_harvest_low_threshold and (pool_harvest_low_threshold < 0 or pool_harvest_low_threshold > 98):
        errormsg = 'Pool harvest low threshold is not in the allowed value range of 0 - 98'
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)
    if snap_harvest_high_threshold and (snap_harvest_high_threshold < 1 or snap_harvest_high_threshold > 99):
        errormsg = 'Snap harvest high threshold is not in the allowed value range of 1 - 99'
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)
    if snap_harvest_low_threshold and (snap_harvest_low_threshold < 0 or snap_harvest_low_threshold > 98):
        errormsg = 'Snap harvest low threshold is not in the allowed value range of 0 - 98'
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)