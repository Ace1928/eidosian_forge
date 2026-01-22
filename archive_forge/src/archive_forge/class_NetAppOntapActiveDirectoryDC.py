from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
class NetAppOntapActiveDirectoryDC:
    """
        Create or delete Active Directory preferred domain controllers
    """

    def __init__(self):
        """
            Initialize the Ontap ActiveDirectoryDC class
        """
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, choices=['present', 'absent'], default='present'), vserver=dict(required=True, type='str'), fqdn=dict(required=True, type='str'), server_ip=dict(required=True, type='str'), skip_config_validation=dict(required=False, type='bool')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = netapp_utils.OntapRestAPI(self.module)
        self.rest_api.fail_if_not_rest_minimum_version('na_ontap_active_directory_domain_controllers', 9, 6)
        self.svm_uuid = None

    def get_active_directory_preferred_domain_controllers_rest(self):
        """
        Retrieves the Active Directory preferred DC configuration of an SVM.
        """
        if self.rest_api.meets_rest_minimum_version(True, 9, 12, 0):
            api = 'protocols/active-directory/%s/preferred-domain-controllers' % self.svm_uuid
            query = {'svm.name': self.parameters['vserver'], 'fqdn': self.parameters['fqdn'], 'server_ip': self.parameters['server_ip'], 'fields': 'server_ip,fqdn'}
            record, error = rest_generic.get_one_record(self.rest_api, api, query)
            if error:
                self.module.fail_json(msg='Error on fetching Active Directory preferred DC configuration of an SVM: %s' % error)
            if record:
                return record
        else:
            api = 'private/cli/vserver/active-directory/preferred-dc'
            query = {'vserver': self.parameters['vserver'], 'domain': self.parameters['fqdn'], 'preferred_dc': self.parameters['server_ip'], 'fields': 'domain,preferred-dc'}
            record, error = rest_generic.get_one_record(self.rest_api, api, query)
            if error:
                self.module.fail_json(msg='Error on fetching Active Directory preferred DC configuration of an SVM using cli: %s' % error)
            if record:
                return {'server_ip': self.na_helper.safe_get(record, ['preferred_dc']), 'fqdn': self.na_helper.safe_get(record, ['domain'])}
        return None

    def create_active_directory_preferred_domain_controllers_rest(self):
        """
        Adds the Active Directory preferred DC configuration to an SVM.
        """
        query = {}
        if self.rest_api.meets_rest_minimum_version(True, 9, 12, 0):
            api = 'protocols/active-directory/%s/preferred-domain-controllers' % self.svm_uuid
            body = {'fqdn': self.parameters['fqdn'], 'server_ip': self.parameters['server_ip']}
            if 'skip_config_validation' in self.parameters:
                query['skip_config_validation'] = self.parameters['skip_config_validation']
        else:
            api = 'private/cli/vserver/active-directory/preferred-dc/add'
            body = {'vserver': self.parameters['vserver'], 'domain': self.parameters['fqdn'], 'preferred_dc': [self.parameters['server_ip']]}
            if 'skip_config_validation' in self.parameters:
                query['skip_config_validation'] = self.parameters['skip_config_validation']
        dummy, error = rest_generic.post_async(self.rest_api, api, body, query)
        if error:
            self.module.fail_json(msg='Error on adding Active Directory preferred DC configuration to an SVM: %s' % error)

    def delete_active_directory_preferred_domain_controllers_rest(self):
        """
        Removes the Active Directory preferred DC configuration from an SVM.
        """
        if self.rest_api.meets_rest_minimum_version(True, 9, 12, 0):
            api = 'protocols/active-directory/%s/preferred-domain-controllers/%s/%s' % (self.svm_uuid, self.parameters['fqdn'], self.parameters['server_ip'])
            record, error = rest_generic.delete_async(self.rest_api, api, None)
        else:
            api = 'private/cli/vserver/active-directory/preferred-dc/remove'
            body = {'vserver': self.parameters['vserver'], 'domain': self.parameters['fqdn'], 'preferred_dc': [self.parameters['server_ip']]}
            dummy, error = rest_generic.delete_async(self.rest_api, api, None, body)
        if error:
            self.module.fail_json(msg='Error on deleting Active Directory preferred DC configuration of an SVM: %s' % error)

    def apply(self):
        self.svm_uuid, dummy = rest_vserver.get_vserver_uuid(self.rest_api, self.parameters['vserver'], self.module, True)
        current = self.get_active_directory_preferred_domain_controllers_rest()
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'create':
                self.create_active_directory_preferred_domain_controllers_rest()
            elif cd_action == 'delete':
                self.delete_active_directory_preferred_domain_controllers_rest()
        result = netapp_utils.generate_result(self.na_helper.changed, cd_action)
        self.module.exit_json(**result)