from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def validate_rest_input_parameters(self, action=None):
    if 'vserver' in self.parameters and self.parameters.get('role') in ['cluster', 'intercluster', 'node-mgmt', 'cluster-mgmt']:
        del self.parameters['vserver']
        self.module.warn('Ignoring vserver with REST for non data SVM.')
    errors = []
    if action == 'create':
        if 'vserver' not in self.parameters and 'ipspace' not in self.parameters:
            errors.append('ipspace name must be provided if scope is cluster, or vserver for svm scope.')
        if self.parameters['interface_type'] == 'fc':
            unsupported_fc_options = ['broadcast_domain', 'dns_domain_name', 'is_dns_update_enabled', 'probe_port', 'subnet_name', 'fail_if_subnet_conflicts']
            used_unsupported_fc_options = [option for option in unsupported_fc_options if option in self.parameters]
            if used_unsupported_fc_options:
                plural = 's' if len(used_unsupported_fc_options) > 1 else ''
                errors.append('%s option%s only supported for IP interfaces: %s, interface_type: %s' % (', '.join(used_unsupported_fc_options), plural, self.parameters.get('interface_name'), self.parameters['interface_type']))
        if self.parameters.get('home_port') and self.parameters.get('broadcast_domain'):
            errors.append('home_port and broadcast_domain are mutually exclusive for creating: %s' % self.parameters.get('interface_name'))
    if self.parameters.get('role') == 'intercluster' and self.parameters.get('protocols') is not None:
        errors.append('Protocol cannot be specified for intercluster role, failed to create interface.')
    if errors:
        self.module.fail_json(msg='Error: %s' % '  '.join(errors))
    ignored_keys = []
    for key in self.parameters.get('ignore_zapi_options', []):
        if key in self.parameters:
            del self.parameters[key]
            ignored_keys.append(key)
    if ignored_keys:
        self.module.warn('Ignoring %s' % ', '.join(ignored_keys))