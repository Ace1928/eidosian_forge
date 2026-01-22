from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def absent_network_acl(self):
    network_acl = self.get_network_acl()
    if network_acl:
        self.result['changed'] = True
        args = {'id': network_acl['id']}
        if not self.module.check_mode:
            res = self.query_api('deleteNetworkACLList', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                self.poll_job(res, 'networkacllist')
    return network_acl