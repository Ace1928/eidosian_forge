import abc
import collections
import urllib
import uuid
from keystoneauth1 import _utils
from keystoneauth1 import access
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import identity
from keystoneauth1 import plugin
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_setting_no_discover_hack(self):
    v2_disc = fixture.V2Discovery(self.V2_URL)
    common_disc = fixture.DiscoveryList(href=self.BASE_URL)
    v2_m = self.stub_url('GET', ['v2.0'], base_url=self.BASE_URL, status_code=200, json=v2_disc)
    common_m = self.stub_url('GET', [], base_url=self.BASE_URL, status_code=300, json=common_disc)
    resp_text = uuid.uuid4().hex
    resp_m = self.stub_url('GET', ['v3', 'path'], base_url=self.BASE_URL, status_code=200, text=resp_text)
    token = fixture.V2Token()
    service = token.add_service(self.IDENTITY)
    service.add_endpoint(public=self.V2_URL, admin=self.V2_URL, internal=self.V2_URL)
    self.stub_url('POST', ['tokens'], base_url=self.V2_URL, json=token)
    v2_auth = identity.V2Password(self.V2_URL, username=uuid.uuid4().hex, password=uuid.uuid4().hex)
    sess = session.Session(auth=v2_auth)
    self.assertFalse(v2_m.called)
    self.assertFalse(common_m.called)
    endpoint = sess.get_endpoint(service_type=self.IDENTITY, version=(3, 0), allow_version_hack=True)
    self.assertEqual(self.V3_URL, endpoint)
    self.assertFalse(v2_m.called)
    self.assertTrue(common_m.called_once)
    endpoint = sess.get_endpoint(service_type=self.IDENTITY, version=(3, 0), allow_version_hack=False)
    self.assertIsNone(endpoint)
    self.assertTrue(v2_m.called_once)
    self.assertTrue(common_m.called_once)
    self.assertRaises(exceptions.EndpointNotFound, sess.get, '/path', endpoint_filter={'service_type': 'identity', 'version': (3, 0), 'allow_version_hack': False})
    self.assertFalse(resp_m.called)
    resp = sess.get('/path', endpoint_filter={'service_type': 'identity', 'version': (3, 0), 'allow_version_hack': True})
    self.assertTrue(resp_m.called_once)
    self.assertEqual(resp_text, resp.text)