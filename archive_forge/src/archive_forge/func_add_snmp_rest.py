from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def add_snmp_rest(self):
    api = 'support/snmp/users'
    self.parameters['authentication_method'] = self.parameters.get('authentication_method', 'community')
    body = {'name': self.parameters['snmp_username'], 'authentication_method': self.parameters['authentication_method']}
    if self.parameters.get('authentication_method') == 'usm' or self.parameters.get('authentication_method') == 'both':
        if self.parameters.get('snmpv3'):
            body['snmpv3'] = self.parameters['snmpv3']
    message, error = self.rest_api.post(api, body)
    if error:
        self.module.fail_json(msg=error)