from oslo_serialization import jsonutils
from mistralclient.api import base as api_base
from mistralclient.api.v2 import executions
from mistralclient.tests.unit.v2 import base
def test_report(self):
    url = self.TEST_URL + URL_TEMPLATE_ID % EXEC['id'] + '/report'
    expected_json = {'root_workflow_execution': {}, 'statistics': {}}
    self.requests_mock.get(url, json=expected_json)
    report = self.executions.get_report(EXEC['id'])
    self.assertDictEqual(expected_json, report)