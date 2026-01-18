from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def update_parameters_rest(self):
    """ expand parameters to match REST returned info
            transform legacy input
        """
    if self.scope == 'svm':
        self.parameters['svm'] = {'name': self.parameters.pop('vserver')}
    servers = self.na_helper.safe_get(self.parameters, ['external', 'servers'])
    if servers:
        self.parameters['external']['servers'] = [{'server': self.add_port(server)} for server in servers if server]
    ip_address = self.parameters.pop('ip_address', None)
    if ip_address:
        ip_address += ':%s' % self.parameters.pop('tcp_port')
        self.parameters['external'] = {'servers': [{'server': ip_address}]}