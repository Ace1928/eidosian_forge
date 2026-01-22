from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
class AnsibleCloudStackVpnConnection(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackVpnConnection, self).__init__(module)
        self.returns = {'dpd': 'dpd', 'esplifetime': 'esp_lifetime', 'esppolicy': 'esp_policy', 'gateway': 'gateway', 'ikepolicy': 'ike_policy', 'ikelifetime': 'ike_lifetime', 'publicip': 'public_ip', 'passive': 'passive', 's2svpngatewayid': 'vpn_gateway_id'}
        self.vpn_customer_gateway = None

    def get_vpn_customer_gateway(self, key=None, identifier=None, refresh=False):
        if not refresh and self.vpn_customer_gateway:
            return self._get_by_key(key, self.vpn_customer_gateway)
        args = {'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id'), 'fetch_list': True}
        vpn_customer_gateway = identifier or self.module.params.get('vpn_customer_gateway')
        vcgws = self.query_api('listVpnCustomerGateways', **args)
        if vcgws:
            for vcgw in vcgws:
                if vpn_customer_gateway.lower() in [vcgw['id'], vcgw['name'].lower()]:
                    self.vpn_customer_gateway = vcgw
                    return self._get_by_key(key, self.vpn_customer_gateway)
        self.fail_json(msg='VPN customer gateway not found: %s' % vpn_customer_gateway)

    def get_vpn_gateway(self, key=None):
        args = {'vpcid': self.get_vpc(key='id'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id')}
        vpn_gateways = self.query_api('listVpnGateways', **args)
        if vpn_gateways:
            return self._get_by_key(key, vpn_gateways['vpngateway'][0])
        elif self.module.params.get('force'):
            if self.module.check_mode:
                return {}
            res = self.query_api('createVpnGateway', **args)
            vpn_gateway = self.poll_job(res, 'vpngateway')
            return self._get_by_key(key, vpn_gateway)
        self.fail_json(msg='VPN gateway not found and not forced to create one')

    def get_vpn_connection(self):
        args = {'vpcid': self.get_vpc(key='id'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id')}
        vpn_conns = self.query_api('listVpnConnections', **args)
        if vpn_conns:
            for vpn_conn in vpn_conns['vpnconnection']:
                if self.get_vpn_customer_gateway(key='id') == vpn_conn['s2scustomergatewayid']:
                    return vpn_conn

    def present_vpn_connection(self):
        vpn_conn = self.get_vpn_connection()
        args = {'s2scustomergatewayid': self.get_vpn_customer_gateway(key='id'), 's2svpngatewayid': self.get_vpn_gateway(key='id'), 'passive': self.module.params.get('passive')}
        if not vpn_conn:
            self.result['changed'] = True
            if not self.module.check_mode:
                res = self.query_api('createVpnConnection', **args)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    vpn_conn = self.poll_job(res, 'vpnconnection')
        return vpn_conn

    def absent_vpn_connection(self):
        vpn_conn = self.get_vpn_connection()
        if vpn_conn:
            self.result['changed'] = True
            args = {'id': vpn_conn['id']}
            if not self.module.check_mode:
                res = self.query_api('deleteVpnConnection', **args)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    self.poll_job(res, 'vpnconnection')
        return vpn_conn

    def get_result(self, resource):
        super(AnsibleCloudStackVpnConnection, self).get_result(resource)
        if resource:
            if 'cidrlist' in resource:
                self.result['cidrs'] = resource['cidrlist'].split(',') or [resource['cidrlist']]
            self.result['force_encap'] = True if resource.get('forceencap') else False
            args = {'key': 'name', 'identifier': resource['s2scustomergatewayid'], 'refresh': True}
            self.result['vpn_customer_gateway'] = self.get_vpn_customer_gateway(**args)
        return self.result