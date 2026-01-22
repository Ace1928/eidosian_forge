from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
class TacacsManager(BaseManager):

    def __init__(self, *args, **kwargs):
        self.module = kwargs.get('module', None)
        self.client = F5RestClient(**self.module.params)
        self.want = self.get_module_parameters(params=self.module.params)
        self.have = self.get_api_parameters()
        self.changes = self.get_usable_changes()

    @property
    def returnables(self):
        return TacacsParameters.returnables

    @property
    def updatables(self):
        return TacacsParameters.updatables

    def get_usable_changes(self, params=None):
        return TacacsUsableChanges(params=params)

    def get_reportable_changes(self, params=None):
        return TacacsReportableChanges(params=params)

    def get_module_parameters(self, params=None):
        return TacacsModuleParameters(params=params)

    def get_api_parameters(self, params=None):
        return TacacsApiParameters(params=params)

    def exists(self):
        errors = [401, 403, 409, 500, 501, 502, 503, 504]
        uri = 'https://{0}:{1}/mgmt/tm/auth/tacacs/~Common~system-auth'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status == 404 or ('code' in response and response['code'] == 404):
            return False
        if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
            return True
        if resp.status in errors or ('code' in response and response['code'] in errors):
            if 'message' in response:
                raise F5ModuleError(response['message'])
            else:
                raise F5ModuleError(resp.content)

    def create(self):
        self._set_changed_options()
        if self.module.check_mode:
            return True
        self.create_on_device()
        if self.want.use_for_auth:
            self.update_auth_source_on_device('tacacs')
        return True

    def update(self):
        self.have = self.read_current_from_device()
        if not self.should_update():
            return False
        if self.module.check_mode:
            return True
        result = False
        if self.update_on_device():
            result = True
        if self.want.use_for_auth and self.changes.auth_source == 'tacacs':
            self.update_auth_source_on_device('tacacs')
            result = True
        return result

    def remove(self):
        if self.module.check_mode:
            return True
        self.update_auth_source_on_device('local')
        self.remove_from_device()
        if self.exists():
            raise F5ModuleError('Failed to delete the resource.')
        return True

    def create_on_device(self):
        params = self.changes.api_params()
        params['name'] = 'system-auth'
        uri = 'https://{0}:{1}/mgmt/tm/auth/tacacs'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.post(uri, json=params)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
            return True
        raise F5ModuleError(resp.content)

    def update_on_device(self):
        params = self.changes.api_params()
        if not params:
            return False
        uri = 'https://{0}:{1}/mgmt/tm/auth/tacacs/~Common~system-auth'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.patch(uri, json=params)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
            return True
        raise F5ModuleError(resp.content)

    def remove_from_device(self):
        uri = 'https://{0}:{1}/mgmt/tm/auth/tacacs/~Common~system-auth'.format(self.client.provider['server'], self.client.provider['server_port'])
        response = self.client.api.delete(uri)
        if response.status in [200, 201]:
            return True
        raise F5ModuleError(response.content)

    def read_current_from_device(self):
        uri = 'https://{0}:{1}/mgmt/tm/auth/tacacs/~Common~system-auth'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
            response['auth_source'] = self.read_current_auth_source_from_device()
            return self.get_api_parameters(params=response)
        raise F5ModuleError(resp.content)