from oslo_serialization import jsonutils
from mistralclient.api import base as api_base
from mistralclient.api.v2 import executions
from mistralclient.tests.unit.v2 import base
def test_update_env(self):
    url = self.TEST_URL + URL_TEMPLATE_ID % EXEC['id']
    self.requests_mock.put(url, json=EXEC)
    body = {'state': EXEC['state'], 'params': {'env': {'k1': 'foobar'}}}
    ex = self.executions.update(EXEC['id'], EXEC['state'], env={'k1': 'foobar'})
    self.assertIsNotNone(ex)
    self.assertDictEqual(executions.Execution(self.executions, EXEC).to_dict(), ex.to_dict())
    self.assertDictEqual(body, self.requests_mock.last_request.json())