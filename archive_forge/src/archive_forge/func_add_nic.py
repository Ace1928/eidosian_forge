from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def add_nic(self):
    self.result['changed'] = True
    args = {'virtualmachineid': self.get_vm(key='id'), 'networkid': self.get_network(key='id'), 'ipaddress': self.module.params.get('ip_address')}
    if not self.module.check_mode:
        res = self.query_api('addNicToVirtualMachine', **args)
        if self.module.params.get('poll_async'):
            vm = self.poll_job(res, 'virtualmachine')
            self.nic = self.get_nic_from_result(result=vm)
    return self.nic