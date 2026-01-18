import re
import uuid
from keystoneauth1 import fixture
from oslo_serialization import jsonutils
from testtools import matchers
from keystoneclient import _discover
from keystoneclient.auth import token_endpoint
from keystoneclient import client
from keystoneclient import discover
from keystoneclient import exceptions
from keystoneclient import session
from keystoneclient.tests.unit import utils
from keystoneclient.v2_0 import client as v2_client
from keystoneclient.v3 import client as v3_client
def test_overriding_stored_kwargs(self):
    self.requests_mock.get(BASE_URL, status_code=300, text=V3_VERSION_LIST)
    self.requests_mock.post('%s/auth/tokens' % V3_URL, text=V3_AUTH_RESPONSE, headers={'X-Subject-Token': V3_TOKEN})
    with self.deprecations.expect_deprecations_here():
        disc = discover.Discover(auth_url=BASE_URL, debug=False, username='foo')
    client = disc.create_client(debug=True, password='bar')
    self.assertIsInstance(client, v3_client.Client)
    self.assertFalse(disc._client_kwargs['debug'])
    self.assertEqual(client.username, 'foo')
    self.assertEqual(client.password, 'bar')