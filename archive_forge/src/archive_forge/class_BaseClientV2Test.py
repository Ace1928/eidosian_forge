from unittest import mock
from mistralclient.api.v2 import client
from mistralclient.tests.unit import base
class BaseClientV2Test(base.BaseClientTest):
    TEST_URL = 'http://mistral.example.com'

    def setUp(self):
        super(BaseClientV2Test, self).setUp()
        with mock.patch('mistralclient.auth.keystone.KeystoneAuthHandler.authenticate', return_value={'session': None}):
            self._client = client.Client(project_name='test', mistral_url=self.TEST_URL)
            self.workbooks = self._client.workbooks
            self.executions = self._client.executions
            self.tasks = self._client.tasks
            self.workflows = self._client.workflows
            self.environments = self._client.environments
            self.action_executions = self._client.action_executions
            self.actions = self._client.actions
            self.services = self._client.services
            self.members = self._client.members