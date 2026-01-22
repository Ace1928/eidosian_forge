from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def _create_vpn_customer_gateway(self, vpn_customer_gateway):
    self.result['changed'] = True
    args = self._common_args()
    if not self.module.check_mode:
        res = self.query_api('createVpnCustomerGateway', **args)
        poll_async = self.module.params.get('poll_async')
        if poll_async:
            vpn_customer_gateway = self.poll_job(res, 'vpncustomergateway')
    return vpn_customer_gateway