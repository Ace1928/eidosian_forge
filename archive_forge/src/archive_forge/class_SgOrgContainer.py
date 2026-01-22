from __future__ import absolute_import, division, print_function
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import SGRestAPI
class SgOrgContainer(object):
    """
    Create, modify and delete StorageGRID Tenant Account
    """

    def __init__(self):
        """
        Parse arguments, setup state variables,
        check parameters and ensure request module is installed
        """
        self.argument_spec = netapp_utils.na_storagegrid_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), name=dict(required=True, type='str'), region=dict(required=False, type='str'), compliance=dict(required=False, type='dict', options=dict(auto_delete=dict(required=False, type='bool'), legal_hold=dict(required=False, type='bool'), retention_period_minutes=dict(required=False, type='int'))), s3_object_lock_enabled=dict(required=False, type='bool'), bucket_versioning_enabled=dict(required=False, type='bool')))
        parameter_map = {'auto_delete': 'autoDelete', 'legal_hold': 'legalHold', 'retention_period_minutes': 'retentionPeriodMinutes'}
        self.module = AnsibleModule(argument_spec=self.argument_spec, mutually_exclusive=[('compliance', 's3_object_lock_enabled')], supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = SGRestAPI(self.module)
        self.rest_api.get_sg_product_version(api_root='org')
        self.data_versioning = {}
        self.data_versioning['versioningSuspended'] = True
        self.data = {}
        self.data['name'] = self.parameters['name']
        self.data['region'] = self.parameters.get('region')
        if self.parameters.get('compliance'):
            self.data['compliance'] = dict(((parameter_map[k], v) for k, v in self.parameters['compliance'].items() if v is not None))
        if self.parameters.get('s3_object_lock_enabled') is not None:
            self.rest_api.fail_if_not_sg_minimum_version('S3 Object Lock', 11, 5)
            self.data['s3ObjectLock'] = dict(enabled=self.parameters['s3_object_lock_enabled'])
        if self.parameters.get('bucket_versioning_enabled') is not None:
            self.rest_api.fail_if_not_sg_minimum_version('Bucket versioning configuration', 11, 6)
            self.data_versioning['versioningEnabled'] = self.parameters['bucket_versioning_enabled']
            if self.data_versioning['versioningEnabled']:
                self.data_versioning['versioningSuspended'] = False

    def get_org_container(self):
        params = {'include': 'compliance,region'}
        response, error = self.rest_api.get('api/v3/org/containers', params=params)
        if error:
            self.module.fail_json(msg=error)
        for container in response['data']:
            if container['name'] == self.parameters['name']:
                return container
        return None

    def create_org_container(self):
        api = 'api/v3/org/containers'
        response, error = self.rest_api.post(api, self.data)
        if error:
            self.module.fail_json(msg=error)
        return response['data']

    def get_org_container_versioning(self):
        api = 'api/v3/org/containers/%s/versioning' % self.parameters['name']
        response, error = self.rest_api.get(api)
        if error:
            self.module.fail_json(msg=error)
        return response['data']

    def update_org_container_versioning(self):
        api = 'api/v3/org/containers/%s/versioning' % self.parameters['name']
        response, error = self.rest_api.put(api, self.data_versioning)
        if error:
            self.module.fail_json(msg=error)
        return response['data']

    def fail_if_global_object_lock_disabled(self):
        api = 'api/v3/org/compliance-global'
        response, error = self.rest_api.get(api)
        if error:
            self.module.fail_json(msg=error)
        if not response['data']['complianceEnabled']:
            self.module.fail_json(msg='Error: Global S3 Object Lock setting is not enabled.')

    def update_org_container_compliance(self):
        api = 'api/v3/org/containers/%s/compliance' % self.parameters['name']
        response, error = self.rest_api.put(api, self.data['compliance'])
        if error:
            self.module.fail_json(msg=error)
        return response['data']

    def delete_org_container(self):
        api = 'api/v3/org/containers/%s' % self.parameters['name']
        response, error = self.rest_api.delete(api, None)
        if error:
            self.module.fail_json(msg=error['text'])

    def apply(self):
        """
        Perform pre-checks, call functions and exit
        """
        versioning_config = None
        update_versioning = False
        org_container = self.get_org_container()
        if org_container and self.parameters.get('bucket_versioning_enabled') is not None:
            versioning_config = self.get_org_container_versioning()
        cd_action = self.na_helper.get_cd_action(org_container, self.parameters)
        if cd_action is None and self.parameters['state'] == 'present':
            update_compliance = False
            if self.parameters.get('compliance') and org_container.get('compliance') != self.data['compliance']:
                update_compliance = True
                self.na_helper.changed = True
            if versioning_config and versioning_config['versioningEnabled'] != self.data_versioning['versioningEnabled']:
                update_versioning = True
                self.na_helper.changed = True
        result_message = ''
        resp_data = org_container
        if self.na_helper.changed:
            if self.module.check_mode:
                pass
            elif cd_action == 'delete':
                self.delete_org_container()
                resp_data = None
                result_message = 'Org Container deleted'
            elif cd_action == 'create':
                if self.parameters.get('s3_object_lock_enabled'):
                    self.fail_if_global_object_lock_disabled()
                resp_data = self.create_org_container()
                if self.parameters.get('bucket_versioning_enabled') is not None:
                    self.update_org_container_versioning()
                result_message = 'Org Container created'
            else:
                if update_compliance:
                    resp_data = self.update_org_container_compliance()
                if update_versioning:
                    self.update_org_container_versioning()
                result_message = 'Org Container updated'
        self.module.exit_json(changed=self.na_helper.changed, msg=result_message, resp=resp_data)