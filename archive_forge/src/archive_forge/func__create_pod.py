from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def _create_pod(self):
    required_params = ['start_ip', 'netmask', 'gateway']
    self.module.fail_on_missing_params(required_params=required_params)
    pod = None
    self.result['changed'] = True
    args = self._get_common_pod_args()
    if not self.module.check_mode:
        res = self.query_api('createPod', **args)
        pod = res['pod']
    return pod