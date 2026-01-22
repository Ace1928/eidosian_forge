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
class CatalogHackTests(utils.TestCase):
    TEST_URL = 'http://keystone.server:5000/v2.0'
    OTHER_URL = 'http://other.server:5000/path'
    IDENTITY = 'identity'
    BASE_URL = 'http://keystone.server:5000/'
    V2_URL = BASE_URL + 'v2.0'
    V3_URL = BASE_URL + 'v3'
    PROJECT_ID = uuid.uuid4().hex

    def test_getting_endpoints(self):
        disc = fixture.DiscoveryList(href=self.BASE_URL)
        self.stub_url('GET', ['/'], base_url=self.BASE_URL, json=disc)
        token = fixture.V2Token()
        service = token.add_service(self.IDENTITY)
        service.add_endpoint(public=self.V2_URL, admin=self.V2_URL, internal=self.V2_URL)
        self.stub_url('POST', ['tokens'], base_url=self.V2_URL, json=token)
        v2_auth = identity.V2Password(self.V2_URL, username=uuid.uuid4().hex, password=uuid.uuid4().hex)
        sess = session.Session(auth=v2_auth)
        endpoint = sess.get_endpoint(service_type=self.IDENTITY, interface='public', version=(3, 0))
        self.assertEqual(self.V3_URL, endpoint)

    def test_returns_original_when_discover_fails(self):
        token = fixture.V2Token()
        service = token.add_service(self.IDENTITY)
        service.add_endpoint(public=self.V2_URL, admin=self.V2_URL, internal=self.V2_URL)
        self.stub_url('POST', ['tokens'], base_url=self.V2_URL, json=token)
        self.stub_url('GET', [], base_url=self.BASE_URL, status_code=404)
        self.stub_url('GET', [], base_url=self.V2_URL, status_code=404)
        v2_auth = identity.V2Password(self.V2_URL, username=uuid.uuid4().hex, password=uuid.uuid4().hex)
        sess = session.Session(auth=v2_auth)
        endpoint = sess.get_endpoint(service_type=self.IDENTITY, interface='public', version=(3, 0))
        self.assertEqual(self.V2_URL, endpoint)

    def test_getting_endpoints_project_id_and_trailing_slash_in_disc_url(self):
        disc = fixture.DiscoveryList(href=self.BASE_URL)
        self.stub_url('GET', ['/'], base_url=self.BASE_URL, json=disc)
        token = fixture.V3Token(project_id=self.PROJECT_ID)
        service = token.add_service(self.IDENTITY)
        service.add_endpoint('public', self.V2_URL + '/')
        service.add_endpoint('admin', self.V2_URL + '/')
        kwargs = {'headers': {'X-Subject-Token': self.TEST_TOKEN}}
        self.stub_url('POST', ['auth', 'tokens'], base_url=self.V3_URL, json=token, **kwargs)
        v3_auth = identity.V3Password(self.V3_URL, username=uuid.uuid4().hex, password=uuid.uuid4().hex)
        sess = session.Session(auth=v3_auth)
        endpoint = sess.get_endpoint(service_type=self.IDENTITY, interface='public', version=(3, 0))
        self.assertEqual(self.V3_URL, endpoint)

    def test_returns_original_skipping_discovery(self):
        token = fixture.V2Token()
        service = token.add_service(self.IDENTITY)
        service.add_endpoint(public=self.V2_URL, admin=self.V2_URL, internal=self.V2_URL)
        self.stub_url('POST', ['tokens'], base_url=self.V2_URL, json=token)
        v2_auth = identity.V2Password(self.V2_URL, username=uuid.uuid4().hex, password=uuid.uuid4().hex)
        sess = session.Session(auth=v2_auth)
        endpoint = sess.get_endpoint(service_type=self.IDENTITY, interface='public', skip_discovery=True, version=(3, 0))
        self.assertEqual(self.V2_URL, endpoint)

    def test_endpoint_override_skips_discovery(self):
        token = fixture.V2Token()
        service = token.add_service(self.IDENTITY)
        service.add_endpoint(public=self.V2_URL, admin=self.V2_URL, internal=self.V2_URL)
        self.stub_url('POST', ['tokens'], base_url=self.V2_URL, json=token)
        v2_auth = identity.V2Password(self.V2_URL, username=uuid.uuid4().hex, password=uuid.uuid4().hex)
        sess = session.Session(auth=v2_auth)
        endpoint = sess.get_endpoint(endpoint_override=self.OTHER_URL, service_type=self.IDENTITY, interface='public', version=(3, 0))
        self.assertEqual(self.OTHER_URL, endpoint)

    def test_endpoint_override_data_runs_discovery(self):
        common_disc = fixture.DiscoveryList(v2=False, v3=False)
        common_disc.add_microversion(href=self.OTHER_URL, id='v2.1', min_version='2.1', max_version='2.35')
        common_m = self.stub_url('GET', base_url=self.OTHER_URL, status_code=200, json=common_disc)
        token = fixture.V2Token()
        service = token.add_service(self.IDENTITY)
        service.add_endpoint(public=self.V2_URL, admin=self.V2_URL, internal=self.V2_URL)
        self.stub_url('POST', ['tokens'], base_url=self.V2_URL, json=token)
        v2_auth = identity.V2Password(self.V2_URL, username=uuid.uuid4().hex, password=uuid.uuid4().hex)
        sess = session.Session(auth=v2_auth)
        data = sess.get_endpoint_data(endpoint_override=self.OTHER_URL, service_type=self.IDENTITY, interface='public', min_version=(2, 0), max_version=(2, discover.LATEST))
        self.assertTrue(common_m.called)
        self.assertEqual(self.OTHER_URL, data.service_url)
        self.assertEqual(self.OTHER_URL, data.catalog_url)
        self.assertEqual(self.OTHER_URL, data.url)
        self.assertEqual((2, 1), data.min_microversion)
        self.assertEqual((2, 35), data.max_microversion)
        self.assertEqual((2, 1), data.api_version)

    def test_forcing_discovery(self):
        v2_disc = fixture.V2Discovery(self.V2_URL)
        common_disc = fixture.DiscoveryList(href=self.BASE_URL)
        v2_m = self.stub_url('GET', ['v2.0'], base_url=self.BASE_URL, status_code=200, json={'version': v2_disc})
        common_m = self.stub_url('GET', [], base_url=self.BASE_URL, status_code=300, json=common_disc)
        token = fixture.V2Token()
        service = token.add_service(self.IDENTITY)
        service.add_endpoint(public=self.V2_URL, admin=self.V2_URL, internal=self.V2_URL)
        self.stub_url('POST', ['tokens'], base_url=self.V2_URL, json=token)
        v2_auth = identity.V2Password(self.V2_URL, username=uuid.uuid4().hex, password=uuid.uuid4().hex)
        sess = session.Session(auth=v2_auth)
        self.assertFalse(v2_m.called)
        self.assertFalse(common_m.called)
        data = sess.get_endpoint_data(service_type=self.IDENTITY, discover_versions=True)
        self.assertTrue(v2_m.called)
        self.assertFalse(common_m.called)
        self.assertEqual(self.V2_URL, data.url)
        self.assertEqual((2, 0), data.api_version)

    def test_forcing_discovery_list_returns_url(self):
        common_disc = fixture.DiscoveryList(href=self.BASE_URL)
        v2_m = self.stub_url('GET', ['v2.0'], base_url=self.BASE_URL, status_code=200, json=common_disc)
        token = fixture.V2Token()
        service = token.add_service(self.IDENTITY)
        service.add_endpoint(public=self.V2_URL, admin=self.V2_URL, internal=self.V2_URL)
        self.stub_url('POST', ['tokens'], base_url=self.V2_URL, json=token)
        v2_auth = identity.V2Password(self.V2_URL, username=uuid.uuid4().hex, password=uuid.uuid4().hex)
        sess = session.Session(auth=v2_auth)
        self.assertFalse(v2_m.called)
        data = sess.get_endpoint_data(service_type=self.IDENTITY, discover_versions=True)
        self.assertTrue(v2_m.called)
        self.assertEqual(self.V2_URL, data.url)
        self.assertEqual((2, 0), data.api_version)

    def test_latest_version_gets_latest_version(self):
        common_disc = fixture.DiscoveryList(href=self.BASE_URL)
        v2_m = self.stub_url('GET', base_url=self.BASE_URL, status_code=200, json=common_disc)
        token = fixture.V2Token()
        service = token.add_service(self.IDENTITY)
        service.add_endpoint(public=self.V2_URL, admin=self.V2_URL, internal=self.V2_URL)
        self.stub_url('POST', ['tokens'], base_url=self.V2_URL, json=token)
        v2_auth = identity.V2Password(self.V2_URL, username=uuid.uuid4().hex, password=uuid.uuid4().hex)
        sess = session.Session(auth=v2_auth)
        self.assertFalse(v2_m.called)
        endpoint = sess.get_endpoint(service_type=self.IDENTITY, version='latest')
        self.assertTrue(v2_m.called)
        self.assertEqual(self.V3_URL, endpoint)
        endpoint = sess.get_endpoint(service_type=self.IDENTITY, max_version='latest')
        self.assertEqual(self.V3_URL, endpoint)
        endpoint = sess.get_endpoint(service_type=self.IDENTITY, min_version='latest')
        self.assertEqual(self.V3_URL, endpoint)
        endpoint = sess.get_endpoint(service_type=self.IDENTITY, min_version='latest', max_version='latest')
        self.assertEqual(self.V3_URL, endpoint)
        self.assertRaises(ValueError, sess.get_endpoint, service_type=self.IDENTITY, min_version='latest', max_version='3.0')

    def test_version_range(self):
        v2_disc = fixture.V2Discovery(self.V2_URL)
        common_disc = fixture.DiscoveryList(href=self.BASE_URL)

        def stub_urls():
            v2_m = self.stub_url('GET', ['v2.0'], base_url=self.BASE_URL, status_code=200, json={'version': v2_disc})
            common_m = self.stub_url('GET', base_url=self.BASE_URL, status_code=200, json=common_disc)
            return (v2_m, common_m)
        v2_m, common_m = stub_urls()
        token = fixture.V2Token()
        service = token.add_service(self.IDENTITY)
        service.add_endpoint(public=self.V2_URL, admin=self.V2_URL, internal=self.V2_URL)
        self.stub_url('POST', ['tokens'], base_url=self.V2_URL, json=token)
        v2_auth = identity.V2Password(self.V2_URL, username=uuid.uuid4().hex, password=uuid.uuid4().hex)
        sess = session.Session(auth=v2_auth)
        self.assertFalse(v2_m.called)
        endpoint = sess.get_endpoint(service_type=self.IDENTITY, min_version='2.0', max_version='3.0')
        self.assertFalse(v2_m.called)
        self.assertTrue(common_m.called)
        self.assertEqual(self.V3_URL, endpoint)
        v2_m, common_m = stub_urls()
        endpoint = sess.get_endpoint(service_type=self.IDENTITY, min_version='1', max_version='2')
        self.assertFalse(v2_m.called)
        self.assertFalse(common_m.called)
        self.assertEqual(self.V2_URL, endpoint)
        v2_m, common_m = stub_urls()
        endpoint = sess.get_endpoint(service_type=self.IDENTITY, min_version='4')
        self.assertFalse(v2_m.called)
        self.assertFalse(common_m.called)
        self.assertIsNone(endpoint)
        v2_m, common_m = stub_urls()
        endpoint = sess.get_endpoint(service_type=self.IDENTITY, min_version='2')
        self.assertFalse(v2_m.called)
        self.assertFalse(common_m.called)
        self.assertEqual(self.V3_URL, endpoint)
        v2_m, common_m = stub_urls()
        self.assertRaises(ValueError, sess.get_endpoint, service_type=self.IDENTITY, version=3, min_version='2')
        self.assertFalse(v2_m.called)
        self.assertFalse(common_m.called)

    def test_get_endpoint_data(self):
        common_disc = fixture.DiscoveryList(v2=False, v3=False)
        common_disc.add_microversion(href=self.OTHER_URL, id='v2.1', min_version='2.1', max_version='2.35')
        common_m = self.stub_url('GET', base_url=self.OTHER_URL, status_code=200, json=common_disc)
        token = fixture.V2Token()
        service = token.add_service('network')
        service.add_endpoint(public=self.OTHER_URL, admin=self.OTHER_URL, internal=self.OTHER_URL)
        self.stub_url('POST', ['tokens'], base_url=self.V2_URL, json=token)
        v2_auth = identity.V2Password(self.V2_URL, username=uuid.uuid4().hex, password=uuid.uuid4().hex)
        sess = session.Session(auth=v2_auth)
        self.assertFalse(common_m.called)
        data = sess.get_endpoint_data(service_type='network', min_version='2.0', max_version='3.0')
        self.assertTrue(common_m.called)
        self.assertEqual(self.OTHER_URL, data.url)
        self.assertEqual((2, 1), data.min_microversion)
        self.assertEqual((2, 35), data.max_microversion)
        self.assertEqual((2, 1), data.api_version)

    def test_get_endpoint_data_compute(self):
        common_disc = fixture.DiscoveryList(v2=False, v3=False)
        common_disc.add_nova_microversion(href=self.OTHER_URL, id='v2.1', min_version='2.1', version='2.35')
        common_m = self.stub_url('GET', base_url=self.OTHER_URL, status_code=200, json=common_disc)
        token = fixture.V2Token()
        service = token.add_service('compute')
        service.add_endpoint(public=self.OTHER_URL, admin=self.OTHER_URL, internal=self.OTHER_URL)
        self.stub_url('POST', ['tokens'], base_url=self.V2_URL, json=token)
        v2_auth = identity.V2Password(self.V2_URL, username=uuid.uuid4().hex, password=uuid.uuid4().hex)
        sess = session.Session(auth=v2_auth)
        self.assertFalse(common_m.called)
        data = sess.get_endpoint_data(service_type='compute', min_version='2.0', max_version='3.0')
        self.assertTrue(common_m.called)
        self.assertEqual(self.OTHER_URL, data.url)
        self.assertEqual((2, 1), data.min_microversion)
        self.assertEqual((2, 35), data.max_microversion)
        self.assertEqual((2, 1), data.api_version)

    def test_getting_endpoints_on_auth_interface(self):
        disc = fixture.DiscoveryList(href=self.BASE_URL)
        self.stub_url('GET', ['/'], base_url=self.BASE_URL, status_code=300, json=disc)
        token = fixture.V2Token()
        service = token.add_service(self.IDENTITY)
        service.add_endpoint(public=self.V2_URL, admin=self.V2_URL, internal=self.V2_URL)
        self.stub_url('POST', ['tokens'], base_url=self.V2_URL, json=token)
        v2_auth = identity.V2Password(self.V2_URL, username=uuid.uuid4().hex, password=uuid.uuid4().hex)
        sess = session.Session(auth=v2_auth)
        endpoint = sess.get_endpoint(interface=plugin.AUTH_INTERFACE, version=(3, 0))
        self.assertEqual(self.V3_URL, endpoint)

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