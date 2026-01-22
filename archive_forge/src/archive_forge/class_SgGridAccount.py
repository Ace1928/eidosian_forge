from __future__ import absolute_import, division, print_function
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import (
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import (
class SgGridAccount(object):
    """
    Create, modify and delete StorageGRID Tenant Account
    """

    def __init__(self):
        """
        Parse arguments, setup state variables,
        check parameters and ensure request module is installed
        """
        self.argument_spec = netapp_utils.na_storagegrid_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), name=dict(required=False, type='str'), description=dict(required=False, type='str'), account_id=dict(required=False, type='str'), protocol=dict(required=False, choices=['s3', 'swift']), management=dict(required=False, type='bool', default=True), use_own_identity_source=dict(required=False, type='bool'), allow_platform_services=dict(required=False, type='bool'), allow_select_object_content=dict(required=False, type='bool'), root_access_group=dict(required=False, type='str'), quota_size=dict(required=False, type='int', default=0), quota_size_unit=dict(default='gb', choices=['bytes', 'b', 'kb', 'mb', 'gb', 'tb', 'pb', 'eb', 'zb', 'yb'], type='str'), password=dict(required=False, type='str', no_log=True), update_password=dict(default='on_create', choices=['on_create', 'always'])))
        self.module = AnsibleModule(argument_spec=self.argument_spec, required_if=[('state', 'present', ['name', 'protocol', 'use_own_identity_source', 'allow_platform_services'])], supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = SGRestAPI(self.module)
        self.rest_api.get_sg_product_version()
        self.data = {}
        self.data['name'] = self.parameters['name']
        self.data['capabilities'] = [self.parameters['protocol']]
        if self.parameters.get('description') is not None:
            self.data['description'] = self.parameters['description']
        if self.parameters.get('password') is not None:
            self.data['password'] = self.parameters['password']
        if self.parameters.get('management'):
            self.data['capabilities'].append('management')
        self.data['policy'] = {}
        if 'use_own_identity_source' in self.parameters:
            self.data['policy']['useAccountIdentitySource'] = self.parameters['use_own_identity_source']
        if 'allow_platform_services' in self.parameters:
            self.data['policy']['allowPlatformServices'] = self.parameters['allow_platform_services']
        if self.parameters.get('root_access_group') is not None:
            self.data['grantRootAccessToGroup'] = self.parameters['root_access_group']
        if self.parameters['quota_size'] > 0:
            self.parameters['quota_size'] = self.parameters['quota_size'] * netapp_utils.POW2_BYTE_MAP[self.parameters['quota_size_unit']]
            self.data['policy']['quotaObjectBytes'] = self.parameters['quota_size']
        elif self.parameters['quota_size'] == 0:
            self.data['policy']['quotaObjectBytes'] = None
        self.pw_change = {}
        if self.parameters.get('password') is not None:
            self.pw_change['password'] = self.parameters['password']
        if 'allow_select_object_content' in self.parameters:
            self.rest_api.fail_if_not_sg_minimum_version('S3 SelectObjectContent API', 11, 6)
            self.data['policy']['allowSelectObjectContent'] = self.parameters['allow_select_object_content']

    def get_tenant_account_id(self):
        api = 'api/v3/grid/accounts'
        params = {'limit': 20}
        params['marker'] = ''
        while params['marker'] is not None:
            list_accounts, error = self.rest_api.get(api, params)
            if error:
                self.module.fail_json(msg=error)
            if len(list_accounts.get('data')) > 0:
                for account in list_accounts['data']:
                    if account['name'] == self.parameters['name']:
                        return account['id']
                params['marker'] = list_accounts['data'][-1]['id']
            else:
                params['marker'] = None
        return None

    def get_tenant_account(self, account_id):
        api = 'api/v3/grid/accounts/%s' % account_id
        account, error = self.rest_api.get(api)
        if error:
            self.module.fail_json(msg=error)
        else:
            return account['data']
        return None

    def create_tenant_account(self):
        api = 'api/v3/grid/accounts'
        response, error = self.rest_api.post(api, self.data)
        if error:
            self.module.fail_json(msg=error)
        return response['data']

    def delete_tenant_account(self, account_id):
        api = 'api/v3/grid/accounts/' + account_id
        self.data = None
        response, error = self.rest_api.delete(api, self.data)
        if error:
            self.module.fail_json(msg=error)

    def update_tenant_account(self, account_id):
        api = 'api/v3/grid/accounts/' + account_id
        if 'password' in self.data:
            del self.data['password']
        if 'grantRootAccessToGroup' in self.data:
            del self.data['grantRootAccessToGroup']
        response, error = self.rest_api.put(api, self.data)
        if error:
            self.module.fail_json(msg=error)
        return response['data']

    def set_tenant_root_password(self, account_id):
        api = 'api/v3/grid/accounts/%s/change-password' % account_id
        response, error = self.rest_api.post(api, self.pw_change)
        if error:
            self.module.fail_json(msg=error['text'])

    def apply(self):
        """
        Perform pre-checks, call functions and exit
        """
        tenant_account = None
        if self.parameters.get('account_id'):
            tenant_account = self.get_tenant_account(self.parameters['account_id'])
        else:
            tenant_account_id = self.get_tenant_account_id()
            if tenant_account_id:
                tenant_account = self.get_tenant_account(tenant_account_id)
        cd_action = self.na_helper.get_cd_action(tenant_account, self.parameters)
        if cd_action is None and self.parameters['state'] == 'present':
            modify = self.na_helper.get_modified_attributes(tenant_account, self.data)
        result_message = ''
        resp_data = tenant_account
        if self.na_helper.changed:
            if self.module.check_mode:
                pass
            elif cd_action == 'delete':
                self.delete_tenant_account(tenant_account['id'])
                result_message = 'Tenant Account deleted'
                resp_data = None
            elif cd_action == 'create':
                resp_data = self.create_tenant_account()
                result_message = 'Tenant Account created'
            elif modify:
                resp_data = self.update_tenant_account(tenant_account['id'])
                result_message = 'Tenant Account updated'
        if self.pw_change:
            if self.module.check_mode:
                pass
            elif self.parameters['update_password'] == 'always' and cd_action != 'create':
                self.set_tenant_root_password(tenant_account['id'])
                self.na_helper.changed = True
                results = [result_message, 'Tenant Account root password updated']
                result_message = '; '.join(filter(None, results))
        self.module.exit_json(changed=self.na_helper.changed, msg=result_message, resp=resp_data)