from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
class AnsibleCloudStackInstanceNic(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackInstanceNic, self).__init__(module)
        self.nic = None
        self.returns = {'ipaddress': 'ip_address', 'macaddress': 'mac_address', 'netmask': 'netmask'}

    def get_nic(self):
        if self.nic:
            return self.nic
        args = {'virtualmachineid': self.get_vm(key='id'), 'networkid': self.get_network(key='id')}
        nics = self.query_api('listNics', **args)
        if nics:
            self.nic = nics['nic'][0]
            return self.nic
        return None

    def get_nic_from_result(self, result):
        for nic in result.get('nic') or []:
            if nic['networkid'] == self.get_network(key='id'):
                return nic

    def add_nic(self):
        self.result['changed'] = True
        args = {'virtualmachineid': self.get_vm(key='id'), 'networkid': self.get_network(key='id'), 'ipaddress': self.module.params.get('ip_address')}
        if not self.module.check_mode:
            res = self.query_api('addNicToVirtualMachine', **args)
            if self.module.params.get('poll_async'):
                vm = self.poll_job(res, 'virtualmachine')
                self.nic = self.get_nic_from_result(result=vm)
        return self.nic

    def update_nic(self, nic):
        ip_address = self.module.params.get('ip_address')
        if not ip_address:
            return nic
        args = {'nicid': nic['id'], 'ipaddress': ip_address}
        if self.has_changed(args, nic, ['ipaddress']):
            self.result['changed'] = True
            if not self.module.check_mode:
                res = self.query_api('updateVmNicIp', **args)
                if self.module.params.get('poll_async'):
                    vm = self.poll_job(res, 'virtualmachine')
                    self.nic = self.get_nic_from_result(result=vm)
        return self.nic

    def remove_nic(self, nic):
        self.result['changed'] = True
        args = {'virtualmachineid': self.get_vm(key='id'), 'nicid': nic['id']}
        if not self.module.check_mode:
            res = self.query_api('removeNicFromVirtualMachine', **args)
            if self.module.params.get('poll_async'):
                self.poll_job(res, 'virtualmachine')
        return nic

    def present_nic(self):
        nic = self.get_nic()
        if not nic:
            nic = self.add_nic()
        else:
            nic = self.update_nic(nic)
        return nic

    def absent_nic(self):
        nic = self.get_nic()
        if nic:
            return self.remove_nic(nic)
        return nic

    def get_result(self, resource):
        super(AnsibleCloudStackInstanceNic, self).get_result(resource)
        if resource and (not self.module.params.get('network')):
            self.module.params['network'] = resource.get('networkid')
        self.result['network'] = self.get_network(key='name')
        self.result['vm'] = self.get_vm(key='name')
        return self.result