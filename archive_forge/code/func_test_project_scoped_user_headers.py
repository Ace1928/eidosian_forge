import itertools
import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from keystonemiddleware.auth_token import _request
from keystonemiddleware.tests.unit import utils
def test_project_scoped_user_headers(self):
    token = fixture.V3Token()
    token.set_project_scope()
    token_id = uuid.uuid4().hex
    auth_ref = access.create(auth_token=token_id, body=token)
    self.request.set_user_headers(auth_ref)
    self._test_v3_headers(token, '')