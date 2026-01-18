from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def update_health_check_config(self, switch_object, health_check_config):
    """Update Health Check config"""
    try:
        task = switch_object.UpdateDVSHealthCheckConfig_Task(healthCheckConfig=health_check_config)
    except vim.fault.DvsFault as dvs_fault:
        self.module.fail_json(msg='Update failed due to DVS fault : %s' % to_native(dvs_fault))
    except vmodl.fault.NotSupported as not_supported:
        self.module.fail_json(msg='Health check not supported on the switch : %s' % to_native(not_supported))
    except TaskError as invalid_argument:
        self.module.fail_json(msg='Failed to configure health check : %s' % to_native(invalid_argument))
    try:
        wait_for_task(task)
    except TaskError as invalid_argument:
        self.module.fail_json(msg='Failed to update health check config : %s' % to_native(invalid_argument))