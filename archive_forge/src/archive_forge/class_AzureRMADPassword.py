from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
import uuid
import datetime
class AzureRMADPassword(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(app_id=dict(type='str'), service_principal_object_id=dict(type='str'), app_object_id=dict(type='str'), key_id=dict(type='str'), tenant=dict(type='str', required=True), value=dict(type='str'), end_date=dict(type='str'), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.state = None
        self.tenant = None
        self.app_id = None
        self.service_principal_object_id = None
        self.app_object_id = None
        self.key_id = None
        self.value = None
        self.end_date = None
        self.results = dict(changed=False)
        self.client = None
        super(AzureRMADPassword, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=False, supports_tags=False, is_ad_resource=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()):
            setattr(self, key, kwargs[key])
        self.client = self.get_graphrbac_client(self.tenant)
        self.resolve_app_obj_id()
        passwords = self.get_all_passwords()
        if self.state == 'present':
            if self.key_id and self.key_exists(passwords):
                self.update(passwords)
            else:
                self.create_password(passwords)
        elif self.key_id is None:
            self.delete_all_passwords(passwords)
        else:
            self.delete_password(passwords)
        return self.results

    def key_exists(self, old_passwords):
        for pd in old_passwords:
            if pd.key_id == self.key_id:
                return True
        return False

    def resolve_app_obj_id(self):
        try:
            if self.app_object_id is not None:
                return
            elif self.app_id or self.service_principal_object_id:
                if not self.app_id:
                    sp = self.client.service_principals.get(self.service_principal_object_id)
                    self.app_id = sp.app_id
                if not self.app_id:
                    self.fail("can't resolve app via service principal object id {0}".format(self.service_principal_object_id))
                result = list(self.client.applications.list(filter="appId eq '{0}'".format(self.app_id)))
                if result:
                    self.app_object_id = result[0].object_id
                else:
                    self.fail("can't resolve app via app id {0}".format(self.app_id))
            else:
                self.fail('one of the [app_id, app_object_id, service_principal_id] must be set')
        except GraphErrorException as ge:
            self.fail('error in resolve app_object_id {0}'.format(str(ge)))

    def get_all_passwords(self):
        try:
            return list(self.client.applications.list_password_credentials(self.app_object_id))
        except GraphErrorException as ge:
            self.fail('failed to fetch passwords for app {0}: {1}'.format(self.app_object_id, str(ge)))

    def delete_all_passwords(self, old_passwords):
        if len(old_passwords) == 0:
            self.results['changed'] = False
            return
        try:
            self.client.applications.patch(self.app_object_id, ApplicationUpdateParameters(password_credentials=[]))
            self.results['changed'] = True
        except GraphErrorException as ge:
            self.fail('fail to purge all passwords for app: {0} - {1}'.format(self.app_object_id, str(ge)))

    def delete_password(self, old_passwords):
        if not self.key_exists(old_passwords):
            self.results['changed'] = False
            return
        num_of_passwords_before_delete = len(old_passwords)
        for pd in old_passwords:
            if pd.key_id == self.key_id:
                old_passwords.remove(pd)
                break
        try:
            self.client.applications.patch(self.app_object_id, ApplicationUpdateParameters(password_credentials=old_passwords))
            num_of_passwords_after_delete = len(self.get_all_passwords())
            if num_of_passwords_after_delete != num_of_passwords_before_delete:
                self.results['changed'] = True
        except GraphErrorException as ge:
            self.fail('failed to delete password with key id {0} - {1}'.format(self.app_id, str(ge)))

    def create_password(self, old_passwords):

        def gen_guid():
            return uuid.uuid4()
        if self.value is None:
            self.fail("when creating a new password, module parameter value can't be None")
        start_date = datetime.datetime.now(datetime.timezone.utc)
        end_date = self.end_date or start_date + relativedelta(years=1)
        value = self.value
        key_id = self.key_id or str(gen_guid())
        new_password = PasswordCredential(start_date=start_date, end_date=end_date, key_id=key_id, value=value, custom_key_identifier=None)
        old_passwords.append(new_password)
        try:
            client = self.get_graphrbac_client(self.tenant)
            app_patch_parameters = ApplicationUpdateParameters(password_credentials=old_passwords)
            client.applications.patch(self.app_object_id, app_patch_parameters)
            new_passwords = self.get_all_passwords()
            for pd in new_passwords:
                if pd.key_id == key_id:
                    self.results['changed'] = True
                    self.results.update(self.to_dict(pd))
        except GraphErrorException as ge:
            self.fail('failed to create new password: {0}'.format(str(ge)))

    def update_password(self, old_passwords):
        self.fail('update existing password is not supported')

    def to_dict(self, pd):
        return dict(end_date=pd.end_date, start_date=pd.start_date, key_id=pd.key_id)