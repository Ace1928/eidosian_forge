from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
class NetAppOntapSecurityIPsecConfig:

    def __init__(self):
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present'], default='present'), enabled=dict(required=False, type='bool'), replay_window=dict(required=False, type='str', choices=['0', '64', '128', '256', '512', '1024'])))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.uuid = None
        self.na_helper = NetAppModule(self.module)
        self.parameters = self.na_helper.check_and_set_parameters(self.module)
        self.rest_api = netapp_utils.OntapRestAPI(self.module)
        self.rest_api.fail_if_not_rest_minimum_version('na_ontap_security_ipsec_config:', 9, 8)

    def get_security_ipsec_config(self):
        """Get IPsec config details"""
        record, error = rest_generic.get_one_record(self.rest_api, 'security/ipsec', None, 'enabled,replay_window')
        if error:
            self.module.fail_json(msg='Error fetching security IPsec config: %s' % to_native(error), exception=traceback.format_exc())
        if record:
            return {'enabled': record.get('enabled'), 'replay_window': record.get('replay_window')}
        return None

    def modify_security_ipsec_config(self, modify):
        """
        Modify security ipsec config
        """
        dummy, error = rest_generic.patch_async(self.rest_api, 'security/ipsec', None, modify)
        if error:
            self.module.fail_json(msg='Error modifying security IPsec config: %s.' % to_native(error), exception=traceback.format_exc())

    def apply(self):
        modify = self.na_helper.get_modified_attributes(self.get_security_ipsec_config(), self.parameters)
        if self.na_helper.changed and (not self.module.check_mode):
            self.modify_security_ipsec_config(modify)
        result = netapp_utils.generate_result(self.na_helper.changed, modify=modify)
        self.module.exit_json(**result)