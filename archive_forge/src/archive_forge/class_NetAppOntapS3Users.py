from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
class NetAppOntapS3Users:

    def __init__(self):
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), vserver=dict(required=True, type='str'), name=dict(required=True, type='str'), comment=dict(required=False, type='str')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.svm_uuid = None
        self.na_helper = NetAppModule(self.module)
        self.parameters = self.na_helper.check_and_set_parameters(self.module)
        self.rest_api = OntapRestAPI(self.module)
        self.use_rest = self.rest_api.is_rest()
        self.rest_api.fail_if_not_rest_minimum_version('na_ontap_s3_users', 9, 8)

    def get_s3_user(self):
        self.get_svm_uuid()
        api = 'protocols/s3/services/%s/users' % self.svm_uuid
        fields = ','.join(('name', 'comment'))
        params = {'name': self.parameters['name'], 'fields': fields}
        record, error = rest_generic.get_one_record(self.rest_api, api, params)
        if error:
            self.module.fail_json(msg='Error fetching S3 user %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
        return record

    def get_svm_uuid(self):
        record, error = rest_vserver.get_vserver_uuid(self.rest_api, self.parameters['vserver'], self.module, True)
        self.svm_uuid = record

    def create_s3_user(self):
        api = 'protocols/s3/services/%s/users' % self.svm_uuid
        body = {'name': self.parameters['name']}
        if self.parameters.get('comment'):
            body['comment'] = self.parameters['comment']
        response, error = rest_generic.post_async(self.rest_api, api, body)
        if error:
            self.module.fail_json(msg='Error creating S3 user %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
        if response.get('num_records') == 1:
            return response.get('records')[0]
        self.module.fail_json(msg='Error creating S3 user %s' % self.parameters['name'], exception=traceback.format_exc())

    def delete_s3_user(self):
        api = 'protocols/s3/services/%s/users' % self.svm_uuid
        dummy, error = rest_generic.delete_async(self.rest_api, api, self.parameters['name'])
        if error:
            self.module.fail_json(msg='Error deleting S3 user %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def modify_s3_user(self, modify):
        api = 'protocols/s3/services/%s/users' % self.svm_uuid
        body = {}
        if modify.get('comment'):
            body['comment'] = self.parameters['comment']
        dummy, error = rest_generic.patch_async(self.rest_api, api, self.parameters['name'], body)
        if error:
            self.module.fail_json(msg='Error modifying S3 user %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def parse_response(self, response):
        if response is not None:
            return (response.get('secret_key'), response.get('access_key'))
        return (None, None)

    def apply(self):
        current = self.get_s3_user()
        cd_action, modify, response = (None, None, None)
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        if cd_action is None:
            modify = self.na_helper.get_modified_attributes(current, self.parameters)
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'create':
                response = self.create_s3_user()
            if cd_action == 'delete':
                self.delete_s3_user()
            if modify:
                self.modify_s3_user(modify)
        secret_key, access_key = self.parse_response(response)
        result = netapp_utils.generate_result(self.na_helper.changed, cd_action, modify, extra_responses={'secret_key': secret_key, 'access_key': access_key})
        self.module.exit_json(**result)