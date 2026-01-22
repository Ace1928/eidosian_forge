from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
class AnsibleCloudStackVpnCustomerGateway(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackVpnCustomerGateway, self).__init__(module)
        self.returns = {'dpd': 'dpd', 'esplifetime': 'esp_lifetime', 'esppolicy': 'esp_policy', 'gateway': 'gateway', 'ikepolicy': 'ike_policy', 'ikelifetime': 'ike_lifetime', 'ipaddress': 'ip_address'}

    def _common_args(self):
        return {'name': self.module.params.get('name'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id'), 'cidrlist': ','.join(self.module.params.get('cidrs')) if self.module.params.get('cidrs') is not None else None, 'esppolicy': self.module.params.get('esp_policy'), 'esplifetime': self.module.params.get('esp_lifetime'), 'ikepolicy': self.module.params.get('ike_policy'), 'ikelifetime': self.module.params.get('ike_lifetime'), 'ipsecpsk': self.module.params.get('ipsec_psk'), 'dpd': self.module.params.get('dpd'), 'forceencap': self.module.params.get('force_encap'), 'gateway': self.module.params.get('gateway')}

    def get_vpn_customer_gateway(self):
        args = {'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id'), 'fetch_list': True}
        vpn_customer_gateway = self.module.params.get('name')
        vpn_customer_gateways = self.query_api('listVpnCustomerGateways', **args)
        if vpn_customer_gateways:
            for vgw in vpn_customer_gateways:
                if vpn_customer_gateway.lower() in [vgw['id'], vgw['name'].lower()]:
                    return vgw

    def present_vpn_customer_gateway(self):
        vpn_customer_gateway = self.get_vpn_customer_gateway()
        required_params = ['cidrs', 'esp_policy', 'gateway', 'ike_policy', 'ipsec_psk']
        self.module.fail_on_missing_params(required_params=required_params)
        if not vpn_customer_gateway:
            vpn_customer_gateway = self._create_vpn_customer_gateway(vpn_customer_gateway)
        else:
            vpn_customer_gateway = self._update_vpn_customer_gateway(vpn_customer_gateway)
        return vpn_customer_gateway

    def _create_vpn_customer_gateway(self, vpn_customer_gateway):
        self.result['changed'] = True
        args = self._common_args()
        if not self.module.check_mode:
            res = self.query_api('createVpnCustomerGateway', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                vpn_customer_gateway = self.poll_job(res, 'vpncustomergateway')
        return vpn_customer_gateway

    def _update_vpn_customer_gateway(self, vpn_customer_gateway):
        args = self._common_args()
        args.update({'id': vpn_customer_gateway['id']})
        if self.has_changed(args, vpn_customer_gateway, skip_diff_for_keys=['ipsecpsk']):
            self.result['changed'] = True
            if not self.module.check_mode:
                res = self.query_api('updateVpnCustomerGateway', **args)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    vpn_customer_gateway = self.poll_job(res, 'vpncustomergateway')
        return vpn_customer_gateway

    def absent_vpn_customer_gateway(self):
        vpn_customer_gateway = self.get_vpn_customer_gateway()
        if vpn_customer_gateway:
            self.result['changed'] = True
            args = {'id': vpn_customer_gateway['id']}
            if not self.module.check_mode:
                res = self.query_api('deleteVpnCustomerGateway', **args)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    self.poll_job(res, 'vpncustomergateway')
        return vpn_customer_gateway

    def get_result(self, resource):
        super(AnsibleCloudStackVpnCustomerGateway, self).get_result(resource)
        if resource:
            if 'cidrlist' in resource:
                self.result['cidrs'] = resource['cidrlist'].split(',') or [resource['cidrlist']]
            self.result['force_encap'] = True if resource.get('forceencap') else False
        return self.result