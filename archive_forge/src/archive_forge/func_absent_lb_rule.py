from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def absent_lb_rule(self):
    args = self._get_common_args()
    rule = self.get_rule(**args)
    if rule:
        self.result['changed'] = True
    if rule and (not self.module.check_mode):
        res = self.query_api('deleteLoadBalancerRule', id=rule['id'])
        poll_async = self.module.params.get('poll_async')
        if poll_async:
            self.poll_job(res, 'loadbalancer')
    return rule