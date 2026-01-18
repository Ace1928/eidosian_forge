from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def update_lacp_group_config(self, switch_object, lacp_group_spec):
    """Update LACP group config"""
    try:
        task = switch_object.UpdateDVSLacpGroupConfig_Task(lacpGroupSpec=lacp_group_spec)
        result = wait_for_task(task)
    except vim.fault.DvsFault as dvs_fault:
        self.module.fail_json(msg='Update failed due to DVS fault : %s' % to_native(dvs_fault))
    except vmodl.fault.NotSupported as not_supported:
        self.module.fail_json(msg='Multiple Link Aggregation Control Protocol groups not supported on the switch : %s' % to_native(not_supported))
    except TaskError as invalid_argument:
        self.module.fail_json(msg='Failed to update Link Aggregation Group : %s' % to_native(invalid_argument))
    return result