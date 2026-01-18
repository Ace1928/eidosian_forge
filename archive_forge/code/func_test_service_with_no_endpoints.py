import uuid
from pycadf import cadftaxonomy as taxonomy
import webob
from keystonemiddleware import audit
from keystonemiddleware.tests.unit.audit import base
def test_service_with_no_endpoints(self):
    env_headers = {'HTTP_X_SERVICE_CATALOG': '[{"endpoints_links": [],\n                             "endpoints": [],\n                             "type": "foo",\n                             "name": "bar"}]', 'HTTP_X_USER_ID': 'user_id', 'HTTP_X_USER_NAME': 'user_name', 'HTTP_X_AUTH_TOKEN': 'token', 'HTTP_X_PROJECT_ID': 'tenant_id', 'HTTP_X_IDENTITY_STATUS': 'Confirmed', 'REQUEST_METHOD': 'GET'}
    url = 'http://public_host:8774/v2/' + str(uuid.uuid4()) + '/servers'
    payload = self.get_payload('GET', url, environ=env_headers)
    self.assertEqual(payload['target']['name'], 'unknown')