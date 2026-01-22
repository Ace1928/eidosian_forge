from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def absent_network_acl_rule(self):
    network_acl_rule = self.get_network_acl_rule()
    if network_acl_rule:
        self.result['changed'] = True
        args = {'id': network_acl_rule['id']}
        if not self.module.check_mode:
            res = self.query_api('deleteNetworkACL', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                self.poll_job(res, 'networkacl')
    return network_acl_rule