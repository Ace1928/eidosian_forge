from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
class AnsibleCloudStackVpnGateway(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackVpnGateway, self).__init__(module)
        self.returns = {'publicip': 'public_ip'}

    def get_vpn_gateway(self):
        args = {'vpcid': self.get_vpc(key='id'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id')}
        vpn_gateways = self.query_api('listVpnGateways', **args)
        if vpn_gateways:
            return vpn_gateways['vpngateway'][0]
        return None

    def present_vpn_gateway(self):
        vpn_gateway = self.get_vpn_gateway()
        if not vpn_gateway:
            self.result['changed'] = True
            args = {'vpcid': self.get_vpc(key='id'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id')}
            if not self.module.check_mode:
                res = self.query_api('createVpnGateway', **args)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    vpn_gateway = self.poll_job(res, 'vpngateway')
        return vpn_gateway

    def absent_vpn_gateway(self):
        vpn_gateway = self.get_vpn_gateway()
        if vpn_gateway:
            self.result['changed'] = True
            args = {'id': vpn_gateway['id']}
            if not self.module.check_mode:
                res = self.query_api('deleteVpnGateway', **args)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    self.poll_job(res, 'vpngateway')
        return vpn_gateway

    def get_result(self, resource):
        super(AnsibleCloudStackVpnGateway, self).get_result(resource)
        if resource:
            self.result['vpc'] = self.get_vpc(key='name')
        return self.result