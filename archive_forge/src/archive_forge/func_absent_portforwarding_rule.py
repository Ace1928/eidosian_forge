from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def absent_portforwarding_rule(self):
    portforwarding_rule = self.get_portforwarding_rule()
    if portforwarding_rule:
        self.result['changed'] = True
        args = {'id': portforwarding_rule['id']}
        if not self.module.check_mode:
            res = self.query_api('deletePortForwardingRule', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                self.poll_job(res, 'portforwardingrule')
    return portforwarding_rule