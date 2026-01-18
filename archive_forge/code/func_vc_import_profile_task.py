from __future__ import absolute_import, division, print_function
import_profile:
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.compat.version import LooseVersion
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi
import json
import time
def vc_import_profile_task(self):
    infra = self.api_client.appliance.infraprofile.Configs
    import_profile_spec = self.get_import_profile_spec()
    import_task = infra.import_profile_task(import_profile_spec)
    self.wait_for_task(import_task)
    if 'SUCCEEDED' == import_task.get_info().status:
        self.module.exit_json(changed=True, status=import_task.get_info().result.value)
    self.module.fail_json(msg='Failed to import profile status:"%s" ' % import_task.get_info().status)