from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def _update_pod(self):
    pod = self.get_pod()
    args = self._get_common_pod_args()
    args['id'] = pod['id']
    if self.has_changed(args, pod):
        self.result['changed'] = True
        if not self.module.check_mode:
            res = self.query_api('updatePod', **args)
            pod = res['pod']
    return pod