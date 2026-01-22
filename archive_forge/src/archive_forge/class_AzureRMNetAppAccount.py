from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import to_native
from ansible_collections.netapp.azure.plugins.module_utils.azure_rm_netapp_common import AzureRMNetAppModuleBase
from ansible_collections.netapp.azure.plugins.module_utils.netapp_module import NetAppModule
class AzureRMNetAppAccount(AzureRMNetAppModuleBase):
    """ create, modify, delete account, including joining AD domain
    """

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), location=dict(type='str', required=False), state=dict(choices=['present', 'absent'], default='present', type='str'), active_directories=dict(type='list', elements='dict', options=dict(active_directory_id=dict(type='str'), dns=dict(type='list', elements='str'), domain=dict(type='str'), site=dict(type='str'), smb_server_name=dict(type='str'), organizational_unit=dict(type='str'), username=dict(type='str'), password=dict(type='str', no_log=True), aes_encryption=dict(type='bool'), ldap_signing=dict(type='bool'), ad_name=dict(type='str'), kdc_ip=dict(type='str'), server_root_ca_certificate=dict(type='str', no_log=True))), debug=dict(type='bool', default=False))
        self.na_helper = NetAppModule()
        self.parameters = dict()
        self.debug = list()
        self.warnings = list()
        super(AzureRMNetAppAccount, self).__init__(derived_arg_spec=self.module_arg_spec, required_if=[('state', 'present', ['location'])], supports_check_mode=True)

    def get_azure_netapp_account(self):
        """
            Returns NetApp Account object for an existing account
            Return None if account does not exist
        """
        try:
            account_get = self.netapp_client.accounts.get(self.parameters['resource_group'], self.parameters['name'])
        except (CloudError, ResourceNotFoundError):
            return None
        account = vars(account_get)
        ads = None
        if account.get('active_directories') is not None:
            ads = list()
            for each_ad in account.get('active_directories'):
                ad_dict = vars(each_ad)
                dns = ad_dict.get('dns')
                if dns is not None:
                    ad_dict['dns'] = sorted(dns.split(','))
                ads.append(ad_dict)
        account['active_directories'] = ads
        return account

    def create_account_request_body(self, modify=None):
        """
            Create an Azure NetApp Account Request Body
            :return: None
        """
        options = dict()
        location = None
        for attr in ('location', 'tags', 'active_directories'):
            value = self.parameters.get(attr)
            if attr == 'location' and modify is None:
                location = value
                continue
            if value is not None:
                if modify is None or attr in modify:
                    if attr == 'active_directories':
                        ads = list()
                        for ad_dict in value:
                            if ad_dict.get('dns') is not None:
                                ad_dict['dns'] = ','.join(ad_dict['dns'])
                            ads.append(ActiveDirectory(**self.na_helper.filter_out_none_entries(ad_dict)))
                        value = ads
                    options[attr] = value
        if modify is None:
            if location is None:
                self.module.fail_json(msg="Error: 'location' is a required parameter")
            return NetAppAccount(location=location, **options)
        return NetAppAccountPatch(**options)

    def create_azure_netapp_account(self):
        """
            Create an Azure NetApp Account
            :return: None
        """
        account_body = self.create_account_request_body()
        try:
            response = self.get_method('accounts', 'create_or_update')(body=account_body, resource_group_name=self.parameters['resource_group'], account_name=self.parameters['name'])
            while response.done() is not True:
                response.result(10)
        except (CloudError, AzureError) as error:
            self.module.fail_json(msg='Error creating Azure NetApp account %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def update_azure_netapp_account(self, modify):
        """
            Create an Azure NetApp Account
            :return: None
        """
        account_body = self.create_account_request_body(modify)
        try:
            response = self.get_method('accounts', 'update')(body=account_body, resource_group_name=self.parameters['resource_group'], account_name=self.parameters['name'])
            while response.done() is not True:
                response.result(10)
        except (CloudError, AzureError) as error:
            self.module.fail_json(msg='Error creating Azure NetApp account %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def delete_azure_netapp_account(self):
        """
            Delete an Azure NetApp Account
            :return: None
        """
        try:
            response = self.get_method('accounts', 'delete')(resource_group_name=self.parameters['resource_group'], account_name=self.parameters['name'])
            while response.done() is not True:
                response.result(10)
        except (CloudError, AzureError) as error:
            self.module.fail_json(msg='Error deleting Azure NetApp account %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def get_changes_in_ads(self, current, desired):
        c_ads = current.get('active_directories')
        d_ads = desired.get('active_directories')
        if not c_ads:
            return (desired.get('active_directories'), None)
        if not d_ads:
            return (None, current.get('active_directories'))
        if len(c_ads) > 1 or len(d_ads) > 1:
            msg = 'Error checking for AD, currently only one AD is supported.'
            if len(c_ads) > 1:
                msg += '  Current: %s.' % str(c_ads)
            if len(d_ads) > 1:
                msg += '  Desired: %s.' % str(d_ads)
            self.module.fail_json(msg='Error checking for AD, currently only one AD is supported')
        changed = False
        d_ad = d_ads[0]
        c_ad = c_ads[0]
        for key, value in c_ad.items():
            if key == 'password':
                if d_ad.get(key) is None:
                    continue
                self.warnings.append("module is not idempotent if 'password:' is present")
            if d_ad.get(key) is None:
                d_ad[key] = value
            elif d_ad.get(key) != value:
                changed = True
                self.debug.append('key: %s, value %s' % (key, value))
        if changed:
            return ([d_ad], None)
        return (None, None)

    def exec_module(self, **kwargs):
        self.fail_when_import_errors(IMPORT_ERRORS, HAS_AZURE_MGMT_NETAPP)
        for key in list(self.module_arg_spec):
            self.parameters[key] = kwargs[key]
        for key in ['tags']:
            if key in kwargs:
                self.parameters[key] = kwargs[key]
        current = self.get_azure_netapp_account()
        modify = None
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        self.debug.append('current: %s' % str(current))
        if current is not None and cd_action is None:
            ads_to_add, ads_to_delete = self.get_changes_in_ads(current, self.parameters)
            self.parameters.pop('active_directories', None)
            if ads_to_add:
                self.parameters['active_directories'] = ads_to_add
            if ads_to_delete:
                self.module.fail_json(msg='Error: API does not support unjoining an AD', debug=self.debug)
            modify = self.na_helper.get_modified_attributes(current, self.parameters)
            if 'tags' in modify:
                dummy, modify['tags'] = self.update_tags(current.get('tags'))
        if self.na_helper.changed:
            if self.module.check_mode:
                pass
            elif cd_action == 'create':
                self.create_azure_netapp_account()
            elif cd_action == 'delete':
                self.delete_azure_netapp_account()
            elif modify:
                self.update_azure_netapp_account(modify)
        results = dict(changed=self.na_helper.changed, modify=modify)
        if self.warnings:
            results['warnings'] = self.warnings
        if self.parameters['debug']:
            results['debug'] = self.debug
        self.module.exit_json(**results)