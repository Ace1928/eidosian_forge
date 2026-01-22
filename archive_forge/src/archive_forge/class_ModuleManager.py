from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import bigiq_version
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
class ModuleManager(object):

    def __init__(self, *args, **kwargs):
        self.module = kwargs.get('module', None)
        self.client = F5RestClient(**self.module.params)
        self.want = ModuleParameters(params=self.module.params)
        self.want.client = self.client
        self.have = ApiParameters()
        self.changes = UsableChanges()

    def _set_changed_options(self):
        changed = {}
        for key in Parameters.returnables:
            if getattr(self.want, key) is not None:
                changed[key] = getattr(self.want, key)
        if changed:
            self.changes = UsableChanges(params=changed)

    def _update_changed_options(self):
        diff = Difference(self.want, self.have)
        updatables = Parameters.updatables
        changed = dict()
        for k in updatables:
            change = diff.compare(k)
            if change is None:
                continue
            elif isinstance(change, dict):
                changed.update(change)
            else:
                changed[k] = change
        if changed:
            self.changes = UsableChanges(params=changed)
            return True
        return False

    def should_update(self):
        result = self._update_changed_options()
        if result:
            return True
        return False

    def check_bigiq_version(self, version):
        if Version(version) >= Version('6.1.0'):
            raise F5ModuleError('Module supports only BIGIQ version 6.0.x or lower.')

    def exec_module(self):
        start = datetime.now().isoformat()
        version = bigiq_version(self.client)
        self.check_bigiq_version(version)
        changed = False
        result = dict()
        state = self.want.state
        if state == 'present':
            changed = self.present()
        elif state == 'absent':
            changed = self.absent()
        reportable = ReportableChanges(params=self.changes.to_return())
        changes = reportable.to_return()
        result.update(**changes)
        result.update(dict(changed=changed))
        self._announce_deprecations(result)
        send_teem(start, self.client, self.module, version)
        return result

    def _announce_deprecations(self, result):
        warnings = result.pop('__warnings', [])
        for warning in warnings:
            self.client.module.deprecate(msg=warning['msg'], version=warning['version'])

    def present(self):
        if self.exists():
            return False
        else:
            return self.create()

    def exists(self):
        uri = "https://{0}:{1}/mgmt/ap/query/v1/tenants/default/reports/AllApplicationsList?$filter=name+eq+'{2}'".format(self.client.provider['server'], self.client.provider['server_port'], self.want.name)
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status == 200 and 'result' in response and ('totalItems' in response['result']) and (response['result']['totalItems'] == 0):
            return False
        return True

    def remove(self):
        if self.module.check_mode:
            return True
        self_link = self.remove_from_device()
        if self.want.wait:
            self.wait_for_apply_template_task(self_link)
            if self.exists():
                raise F5ModuleError('Failed to delete the resource.')
        return True

    def has_no_service_environment(self):
        if self.want.default_device_reference is None and self.want.ssg_reference is None:
            return True
        return False

    def create(self):
        if self.want.service_environment is None:
            raise F5ModuleError("A 'service_environment' must be specified when creating a new application.")
        if self.want.servers is None:
            raise F5ModuleError("At least one 'servers' item is needed when creating a new application.")
        if self.want.inbound_virtual is None:
            raise F5ModuleError("An 'inbound_virtual' must be specified when creating a new application.")
        self._set_changed_options()
        if self.has_no_service_environment():
            raise F5ModuleError("The specified 'service_environment' ({0}) was not found.".format(self.want.service_environment))
        if self.module.check_mode:
            return True
        self_link = self.create_on_device()
        if self.want.wait:
            self.wait_for_apply_template_task(self_link)
            if not self.exists():
                raise F5ModuleError('Failed to deploy application.')
        return True

    def create_on_device(self):
        params = self.changes.api_params()
        params['mode'] = 'CREATE'
        uri = 'https://{0}:{1}/mgmt/cm/global/tasks/apply-template'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.post(uri, json=params)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if 'code' in response and response['code'] == 400:
            if 'message' in response:
                raise F5ModuleError(response['message'])
            else:
                raise F5ModuleError(resp._content)
        return response['selfLink']

    def absent(self):
        if self.exists():
            return self.remove()
        return False

    def remove_from_device(self):
        params = dict(configSetName=self.want.name, mode='DELETE')
        uri = 'https://{0}:{1}/mgmt/cm/global/tasks/apply-template'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.post(uri, json=params)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if 'code' in response and response['code'] == 400:
            if 'message' in response:
                raise F5ModuleError(response['message'])
            else:
                raise F5ModuleError(resp._content)
        return response['selfLink']

    def wait_for_apply_template_task(self, self_link):
        host = 'https://{0}:{1}'.format(self.client.provider['server'], self.client.provider['server_port'])
        uri = self_link.replace('https://localhost', host)
        while True:
            resp = self.client.api.get(uri)
            try:
                response = resp.json()
            except ValueError as ex:
                raise F5ModuleError(str(ex))
            if response['status'] == 'FINISHED' and response.get('currentStep', None) == 'DONE':
                return True
            elif 'errorMessage' in response:
                raise F5ModuleError(response['errorMessage'])
            time.sleep(5)