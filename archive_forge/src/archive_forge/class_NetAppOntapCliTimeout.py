from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
class NetAppOntapCliTimeout:

    def __init__(self):
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present'], default='present'), timeout=dict(required=True, type='int')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.na_helper = NetAppModule(self.module)
        self.parameters = self.na_helper.check_and_set_parameters(self.module)
        self.rest_api = netapp_utils.OntapRestAPI(self.module)
        self.rest_api.fail_if_not_rest_minimum_version('na_ontap_cli_timeout:', 9, 6)

    def get_timeout_value_rest(self):
        """ Get CLI inactivity timeout value """
        fields = 'timeout'
        api = 'private/cli/system/timeout'
        record, error = rest_generic.get_one_record(self.rest_api, api, query=None, fields=fields)
        if error:
            self.module.fail_json(msg='Error fetching CLI sessions timeout value: %s' % to_native(error), exception=traceback.format_exc())
        if record:
            return {'timeout': record.get('timeout')}
        return None

    def modify_timeout_value_rest(self, modify):
        """ Modify CLI inactivity timeout value """
        api = 'private/cli/system/timeout'
        dummy, error = rest_generic.patch_async(self.rest_api, api, uuid_or_name=None, body=modify)
        if error:
            self.module.fail_json(msg='Error modifying CLI sessions timeout value: %s.' % to_native(error), exception=traceback.format_exc())

    def apply(self):
        current = self.get_timeout_value_rest()
        modify = self.na_helper.get_modified_attributes(current, self.parameters)
        if self.na_helper.changed and (not self.module.check_mode):
            self.modify_timeout_value_rest(modify)
        result = netapp_utils.generate_result(self.na_helper.changed, modify=modify)
        self.module.exit_json(**result)