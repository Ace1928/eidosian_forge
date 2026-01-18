import itertools
import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from keystonemiddleware.auth_token import _request
from keystonemiddleware.tests.unit import utils
def test_removing_headers(self):
    GOOD = ('X-Auth-Token', 'unknownstring', uuid.uuid4().hex)
    BAD = ('X-Domain-Id', 'X-Domain-Name', 'X-Project-Id', 'X-Project-Name', 'X-Project-Domain-Id', 'X-Project-Domain-Name', 'X-User-Id', 'X-User-Name', 'X-User-Domain-Id', 'X-User-Domain-Name', 'X-Roles', 'X-Identity-Status', 'X-Service-Domain-Id', 'X-Service-Domain-Name', 'X-Service-Project-Id', 'X-Service-Project-Name', 'X-Service-Project-Domain-Id', 'X-Service-Project-Domain-Name', 'X-Service-User-Id', 'X-Service-User-Name', 'X-Service-User-Domain-Id', 'X-Service-User-Domain-Name', 'X-Service-Roles', 'X-Service-Identity-Status', 'X-Service-Catalog', 'X-Role', 'X-User', 'X-Tenant-Id', 'X-Tenant-Name', 'X-Tenant')
    header_vals = {}
    for header in itertools.chain(GOOD, BAD):
        v = uuid.uuid4().hex
        header_vals[header] = v
        self.request.headers[header] = v
    self.request.remove_auth_headers()
    for header in BAD:
        self.assertNotIn(header, self.request.headers)
    for header in GOOD:
        self.assertEqual(header_vals[header], self.request.headers[header])