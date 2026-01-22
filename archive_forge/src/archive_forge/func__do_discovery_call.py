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
def _do_discovery_call(self, token=None, **kwargs):
    self.requests_mock.get(BASE_URL, status_code=300, text=V3_VERSION_LIST)
    if not token:
        token = uuid.uuid4().hex
    url = 'http://testurl'
    with self.deprecations.expect_deprecations_here():
        a = token_endpoint.Token(url, token)
        s = session.Session(auth=a)
    discover.Discover(s, auth_url=BASE_URL, **kwargs)
    self.assertEqual(BASE_URL, self.requests_mock.last_request.url)