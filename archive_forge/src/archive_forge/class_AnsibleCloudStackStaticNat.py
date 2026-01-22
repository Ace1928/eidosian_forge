from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
class AnsibleCloudStackStaticNat(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackStaticNat, self).__init__(module)
        self.returns = {'virtualmachinedisplayname': 'vm_display_name', 'virtualmachinename': 'vm_name', 'ipaddress': 'ip_address', 'vmipaddress': 'vm_guest_ip'}

    def create_static_nat(self, ip_address):
        self.result['changed'] = True
        args = {'virtualmachineid': self.get_vm(key='id'), 'ipaddressid': ip_address['id'], 'vmguestip': self.get_vm_guest_ip(), 'networkid': self.get_network(key='id')}
        if not self.module.check_mode:
            self.query_api('enableStaticNat', **args)
            self.ip_address = None
            ip_address = self.get_ip_address()
        return ip_address

    def update_static_nat(self, ip_address):
        args = {'virtualmachineid': self.get_vm(key='id'), 'ipaddressid': ip_address['id'], 'vmguestip': self.get_vm_guest_ip(), 'networkid': self.get_network(key='id')}
        ip_address['vmguestip'] = ip_address['vmipaddress']
        if self.has_changed(args, ip_address, ['vmguestip', 'virtualmachineid']):
            self.result['changed'] = True
            if not self.module.check_mode:
                res = self.query_api('disableStaticNat', ipaddressid=ip_address['id'])
                self.poll_job(res, 'staticnat')
                self.query_api('enableStaticNat', **args)
                self.ip_address = None
                ip_address = self.get_ip_address()
        return ip_address

    def present_static_nat(self):
        ip_address = self.get_ip_address()
        if not ip_address['isstaticnat']:
            ip_address = self.create_static_nat(ip_address)
        else:
            ip_address = self.update_static_nat(ip_address)
        return ip_address

    def absent_static_nat(self):
        ip_address = self.get_ip_address()
        if ip_address['isstaticnat']:
            self.result['changed'] = True
            if not self.module.check_mode:
                res = self.query_api('disableStaticNat', ipaddressid=ip_address['id'])
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    self.poll_job(res, 'staticnat')
        return ip_address