from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def validate_ipsec(self):
    """
        validate ipsec options.
        """
    for end_point in ['local_endpoint', 'remote_endpoint']:
        if self.parameters.get(end_point):
            self.parameters[end_point]['address'] = netapp_ipaddress.validate_and_compress_ip_address(self.parameters[end_point]['address'], self.module)
            self.parameters[end_point]['netmask'] = str(netapp_ipaddress.netmask_to_netmask_length(self.parameters[end_point]['address'], self.parameters[end_point]['netmask'], self.module))
            if self.parameters[end_point].get('port') and '-' not in self.parameters[end_point]['port']:
                self.parameters[end_point]['port'] = self.parameters[end_point]['port'] + '-' + self.parameters[end_point]['port']
    if self.parameters.get('action') in ['bypass', 'discard'] and self.parameters.get('authentication_method') != 'none':
        msg = 'The IPsec action is %s, which does not provide packet protection. The authentication_method and ' % self.parameters['action']
        self.parameters.pop('authentication_method', None)
        if self.parameters.get('secret_key'):
            del self.parameters['secret_key']
            self.module.warn(msg + 'secret_key options are ignored')
        if self.parameters.get('certificate'):
            del self.parameters['certificate']
            self.module.warn(msg + 'certificate options are ignored')
    protocols_info = {'6': 'tcp', '17': 'udp', '0': 'any'}
    if self.parameters.get('protocol') in protocols_info:
        self.parameters['protocol'] = protocols_info[self.parameters['protocol']]