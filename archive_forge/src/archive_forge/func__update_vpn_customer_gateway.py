from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
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