import os.path
import pkg_resources as pkg
from urllib import parse
from urllib import request
from mistralclient.api.v2 import workflows
from mistralclient.tests.unit.v2 import base
class TestWorkflowsV2(base.BaseClientV2Test):

    def test_create(self):
        self.requests_mock.post(self.TEST_URL + URL_TEMPLATE_SCOPE, json={'workflows': [WORKFLOW]}, status_code=201)
        wfs = self.workflows.create(WF_DEF)
        self.assertIsNotNone(wfs)
        self.assertEqual(WF_DEF, wfs[0].definition)
        last_request = self.requests_mock.last_request
        self.assertEqual(WF_DEF, last_request.text)
        self.assertEqual('text/plain', last_request.headers['content-type'])

    def test_create_with_file(self):
        self.requests_mock.post(self.TEST_URL + URL_TEMPLATE_SCOPE, json={'workflows': [WORKFLOW]}, status_code=201)
        path = pkg.resource_filename('mistralclient', 'tests/unit/resources/wf_v2.yaml')
        wfs = self.workflows.create(path)
        self.assertIsNotNone(wfs)
        self.assertEqual(WF_DEF, wfs[0].definition)
        last_request = self.requests_mock.last_request
        self.assertEqual(WF_DEF, last_request.text)
        self.assertEqual('text/plain', last_request.headers['content-type'])

    def test_update(self):
        self.requests_mock.put(self.TEST_URL + URL_TEMPLATE_SCOPE, json={'workflows': [WORKFLOW]})
        wfs = self.workflows.update(WF_DEF)
        self.assertIsNotNone(wfs)
        self.assertEqual(WF_DEF, wfs[0].definition)
        last_request = self.requests_mock.last_request
        self.assertEqual(WF_DEF, last_request.text)
        self.assertEqual('text/plain', last_request.headers['content-type'])

    def test_update_with_id(self):
        self.requests_mock.put(self.TEST_URL + URL_TEMPLATE_NAME % '123', json=WORKFLOW)
        wf = self.workflows.update(WF_DEF, id='123')
        self.assertIsNotNone(wf)
        self.assertEqual(WF_DEF, wf.definition)
        last_request = self.requests_mock.last_request
        self.assertEqual('namespace=&scope=private', last_request.query)
        self.assertEqual(WF_DEF, last_request.text)
        self.assertEqual('text/plain', last_request.headers['content-type'])

    def test_update_with_file_uri(self):
        self.requests_mock.put(self.TEST_URL + URL_TEMPLATE_SCOPE, json={'workflows': [WORKFLOW]})
        path = pkg.resource_filename('mistralclient', 'tests/unit/resources/wf_v2.yaml')
        path = os.path.abspath(path)
        uri = parse.urljoin('file:', request.pathname2url(path))
        wfs = self.workflows.update(uri)
        self.assertIsNotNone(wfs)
        self.assertEqual(WF_DEF, wfs[0].definition)
        last_request = self.requests_mock.last_request
        self.assertEqual(WF_DEF, last_request.text)
        self.assertEqual('text/plain', last_request.headers['content-type'])

    def test_list(self):
        self.requests_mock.get(self.TEST_URL + URL_TEMPLATE, json={'workflows': [WORKFLOW]})
        workflows_list = self.workflows.list()
        self.assertEqual(1, len(workflows_list))
        wf = workflows_list[0]
        self.assertEqual(workflows.Workflow(self.workflows, WORKFLOW).to_dict(), wf.to_dict())

    def test_list_with_pagination(self):
        self.requests_mock.get(self.TEST_URL + URL_TEMPLATE, json={'workflows': [WORKFLOW], 'next': '/workflows?fake'})
        workflows_list = self.workflows.list(limit=1, sort_keys='created_at', sort_dirs='asc')
        self.assertEqual(1, len(workflows_list))
        last_request = self.requests_mock.last_request
        self.assertEqual(['1'], last_request.qs['limit'])
        self.assertEqual(['created_at'], last_request.qs['sort_keys'])
        self.assertEqual(['asc'], last_request.qs['sort_dirs'])

    def test_list_with_no_limit(self):
        self.requests_mock.get(self.TEST_URL + URL_TEMPLATE, json={'workflows': [WORKFLOW]})
        workflows_list = self.workflows.list(limit=-1)
        self.assertEqual(1, len(workflows_list))
        last_request = self.requests_mock.last_request
        self.assertNotIn('limit', last_request.qs)

    def test_get(self):
        url = self.TEST_URL + URL_TEMPLATE_NAME % 'wf'
        self.requests_mock.get(url, json=WORKFLOW)
        wf = self.workflows.get('wf')
        self.assertIsNotNone(wf)
        self.assertEqual(workflows.Workflow(self.workflows, WORKFLOW).to_dict(), wf.to_dict())

    def test_delete(self):
        self.requests_mock.delete(self.TEST_URL + URL_TEMPLATE_NAME % 'wf', status_code=204)
        self.workflows.delete('wf')