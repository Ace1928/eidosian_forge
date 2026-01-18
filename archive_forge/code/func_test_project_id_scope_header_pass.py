import uuid
from keystoneauth1 import exceptions
from keystoneauth1.identity.v3 import tokenless_auth
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_project_id_scope_header_pass(self):
    project_id = uuid.uuid4().hex
    auth, session = self.create(auth_url=self.TEST_URL, project_id=project_id)
    session.get(self.TEST_URL, authenticated=True)
    self.assertRequestHeaderEqual('X-Project-Id', project_id)