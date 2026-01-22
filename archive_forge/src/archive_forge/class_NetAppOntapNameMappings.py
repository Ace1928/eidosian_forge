from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
class NetAppOntapNameMappings:
    """ object initialize and class methods """

    def __init__(self):
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), vserver=dict(required=True, type='str'), client_match=dict(required=False, type='str'), direction=dict(required=True, type='str', choices=['krb_unix', 'win_unix', 'unix_win', 's3_unix', 's3_win']), index=dict(required=True, type='int'), from_index=dict(required=False, type='int'), pattern=dict(required=False, type='str'), replacement=dict(required=False, type='str')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.na_helper = NetAppModule(self)
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.svm_uuid = None
        self.rest_api = netapp_utils.OntapRestAPI(self.module)
        self.rest_api.fail_if_not_rest_minimum_version('na_ontap_name_mappings', 9, 6)
        self.rest_api.is_rest_supported_properties(self.parameters, None, [['from_index', (9, 7)]])
        if self.parameters['direction'] in ['s3_unix', 's3_win'] and (not self.rest_api.meets_rest_minimum_version(True, 9, 12, 1)):
            self.module.fail_json(msg='Error: direction %s requires ONTAP 9.12.1 or later version.' % self.parameters['direction'])

    def get_name_mappings_rest(self, index=None):
        """
        Retrieves the name mapping configuration for SVM with rest API.
        """
        if index is None:
            index = self.parameters['index']
        query = {'svm.name': self.parameters.get('vserver'), 'index': index, 'direction': self.parameters.get('direction'), 'fields': 'svm.uuid,client_match,direction,index,pattern,replacement,'}
        api = 'name-services/name-mappings'
        record, error = rest_generic.get_one_record(self.rest_api, api, query)
        if error:
            self.module.fail_json(msg=error)
        if record:
            self.svm_uuid = record['svm']['uuid']
            return {'pattern': self.na_helper.safe_get(record, ['pattern']), 'direction': self.na_helper.safe_get(record, ['direction']), 'replacement': self.na_helper.safe_get(record, ['replacement']), 'client_match': record.get('client_match', None)}
        return None

    def create_name_mappings_rest(self):
        """
        Creates name mappings for an SVM with REST API.
        """
        body = {'svm.name': self.parameters.get('vserver'), 'index': self.parameters.get('index'), 'direction': self.parameters.get('direction'), 'pattern': self.parameters.get('pattern'), 'replacement': self.parameters.get('replacement')}
        if 'client_match' in self.parameters:
            body['client_match'] = self.parameters['client_match']
        api = 'name-services/name-mappings'
        dummy, error = rest_generic.post_async(self.rest_api, api, body)
        if error is not None:
            self.module.fail_json(msg='Error on creating name mappings rest: %s' % error)

    def modify_name_mappings_rest(self, modify=None, reindex=False):
        """
        Updates the name mapping configuration of an SVM with rest API.
        Swap the position with new position(new_index).
        """
        body = {}
        query = None
        if modify:
            for option in ['pattern', 'replacement', 'client_match']:
                if option in modify:
                    body[option] = self.parameters[option]
        index = self.parameters['index']
        if reindex:
            query = {'new_index': self.parameters.get('index')}
            index = self.parameters['from_index']
        api = 'name-services/name-mappings/%s/%s/%s' % (self.svm_uuid, self.parameters['direction'], index)
        dummy, error = rest_generic.patch_async(self.rest_api, api, None, body, query)
        if error is not None:
            self.module.fail_json(msg='Error on modifying name mappings rest: %s' % error)

    def delete_name_mappings_rest(self):
        """
        Delete the name mapping configuration of an SVM with rest API.
        """
        api = 'name-services/name-mappings/%s/%s/%s' % (self.svm_uuid, self.parameters['direction'], self.parameters['index'])
        dummy, error = rest_generic.delete_async(self.rest_api, api, None)
        if error is not None:
            self.module.fail_json(msg='Error on deleting name mappings rest: %s' % error)

    def apply(self):
        reindex = False
        current = self.get_name_mappings_rest()
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        if cd_action == 'create':
            if self.parameters.get('from_index') is not None:
                current = self.get_name_mappings_rest(self.parameters['from_index'])
                if not current:
                    self.module.fail_json(msg='Error from_index entry does not exist')
                reindex = True
                cd_action = None
            elif not self.parameters.get('pattern') or not self.parameters.get('replacement'):
                self.module.fail_json(msg='Error creating name mappings for an SVM, pattern and replacement are required in create.')
        modify = self.na_helper.get_modified_attributes(current, self.parameters) if cd_action is None else None
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'create':
                self.create_name_mappings_rest()
            elif cd_action == 'delete':
                self.delete_name_mappings_rest()
            elif modify or reindex:
                self.modify_name_mappings_rest(modify, reindex)
                if reindex:
                    modify['new_index'] = self.parameters.get('index')
                    modify['from_index'] = self.parameters['from_index']
        result = netapp_utils.generate_result(self.na_helper.changed, cd_action, modify)
        self.module.exit_json(**result)