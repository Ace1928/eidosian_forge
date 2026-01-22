from unittest import mock
from oslotest import base as test_base
from ironicclient.common.apiclient import base
class HumanResourceManager(base.ManagerWithFind):
    resource_class = HumanResource

    def list(self):
        return self._list('/human_resources', 'human_resources')

    def get(self, human_resource):
        return self._get('/human_resources/%s' % base.getid(human_resource), 'human_resource')

    def update(self, human_resource, name):
        body = {'human_resource': {'name': name}}
        return self._put('/human_resources/%s' % base.getid(human_resource), body, 'human_resource')