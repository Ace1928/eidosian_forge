import json
import re
from unittest import mock
from testtools import matchers
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import http_basic
from keystoneauth1 import noauth
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
class VersionDataTests(utils.TestCase):

    def setUp(self):
        super(VersionDataTests, self).setUp()
        self.session = session.Session()

    def test_version_data_basics(self):
        examples = {'keystone': V3_VERSION_LIST, 'cinder': CINDER_EXAMPLES, 'glance': GLANCE_EXAMPLES}
        for path, data in examples.items():
            url = '%s%s' % (BASE_URL, path)
            mock = self.requests_mock.get(url, status_code=300, json=data)
            disc = discover.Discover(self.session, url)
            raw_data = disc.raw_version_data()
            clean_data = disc.version_data()
            for v in raw_data:
                for n in ('id', 'status', 'links'):
                    msg = '%s missing from %s version data' % (n, path)
                    self.assertThat(v, matchers.Annotate(msg, matchers.Contains(n)))
            for v in clean_data:
                for n in ('version', 'url', 'raw_status'):
                    msg = '%s missing from %s version data' % (n, path)
                    self.assertThat(v, matchers.Annotate(msg, matchers.Contains(n)))
            self.assertTrue(mock.called_once)

    def test_version_data_override_version_url(self):
        self.requests_mock.get(V3_URL, status_code=200, json={'version': fixture.V3Discovery('http://override/identity/v3')})
        disc = discover.Discover(self.session, V3_URL)
        version_data = disc.version_data()
        for v in version_data:
            self.assertEqual(v['version'], (3, 0))
            self.assertEqual(v['status'], discover.Status.CURRENT)
            self.assertEqual(v['raw_status'], 'stable')
            self.assertEqual(v['url'], V3_URL)
        self.requests_mock.get(BASE_URL, status_code=200, json={'version': fixture.V3Discovery('http://override/identity/v3')})
        disc = discover.Discover(self.session, BASE_URL)
        version_data = disc.version_data()
        for v in version_data:
            self.assertEqual(v['version'], (3, 0))
            self.assertEqual(v['status'], discover.Status.CURRENT)
            self.assertEqual(v['raw_status'], 'stable')
            self.assertEqual(v['url'], V3_URL)

    def test_version_data_unknown(self):
        discovery_fixture = fixture.V3Discovery(V3_URL)
        discovery_fixture.status = 'hungry'
        discovery_doc = _create_single_version(discovery_fixture)
        self.requests_mock.get(V3_URL, status_code=200, json=discovery_doc)
        disc = discover.Discover(self.session, V3_URL)
        clean_data = disc.version_data(allow_unknown=True)
        self.assertEqual(discover.Status.UNKNOWN, clean_data[0]['status'])

    def test_version_data_individual(self):
        mock = self.requests_mock.get(V3_URL, status_code=200, json=V3_VERSION_ENTRY)
        disc = discover.Discover(self.session, V3_URL)
        raw_data = disc.raw_version_data()
        clean_data = disc.version_data()
        for v in raw_data:
            self.assertEqual(v['id'], 'v3.0')
            self.assertEqual(v['status'], 'stable')
            self.assertIn('media-types', v)
            self.assertIn('links', v)
        for v in clean_data:
            self.assertEqual(v['version'], (3, 0))
            self.assertEqual(v['status'], discover.Status.CURRENT)
            self.assertEqual(v['raw_status'], 'stable')
            self.assertEqual(v['url'], V3_URL)
        self.assertTrue(mock.called_once)

    def test_version_data_legacy_ironic_no_override(self):
        """Validate detection of legacy Ironic microversion ranges."""
        ironic_url = 'https://bare-metal.example.com/v1/'
        self.requests_mock.get(ironic_url, status_code=200, json={'id': 'v1', 'links': [{'href': ironic_url, 'rel': 'self'}]}, headers={'X-OpenStack-Ironic-API-Minimum-Version': '1.3', 'X-OpenStack-Ironic-API-Maximum-Version': '1.21'})
        plugin = noauth.NoAuth()
        a = adapter.Adapter(self.session, auth=plugin, service_type='baremetal')
        self.assertIsNone(a.get_api_major_version())

    def test_version_data_ironic_microversions(self):
        """Validate detection of Ironic microversion ranges."""
        ironic_url = 'https://bare-metal.example.com/v1/'
        self.requests_mock.get(ironic_url, status_code=200, json={'id': 'v1', 'version': {'id': 'v1', 'links': [{'href': ironic_url, 'rel': 'self'}], 'version': '1.40', 'min_version': '1.10', 'status': 'CURRENT'}, 'links': [{'href': ironic_url, 'rel': 'self'}]}, headers={'X-OpenStack-Ironic-API-Minimum-Version': '1.3', 'X-OpenStack-Ironic-API-Maximum-Version': '1.21'})
        self.assertEqual([{'collection': None, 'version': (1, 0), 'url': ironic_url, 'status': discover.Status.CURRENT, 'raw_status': discover.Status.CURRENT, 'min_microversion': (1, 10), 'max_microversion': (1, 40), 'next_min_version': None, 'not_before': None}], discover.Discover(self.session, ironic_url).version_data())

    def test_version_data_legacy_ironic_microversions(self):
        """Validate detection of legacy Ironic microversion ranges."""
        ironic_url = 'https://bare-metal.example.com/v1/'
        self.requests_mock.get(ironic_url, status_code=200, json={'id': 'v1', 'links': [{'href': ironic_url, 'rel': 'self'}]}, headers={'X-OpenStack-Ironic-API-Minimum-Version': '1.3', 'X-OpenStack-Ironic-API-Maximum-Version': '1.21'})
        self.assertEqual([{'collection': None, 'version': (1, 0), 'url': ironic_url, 'status': discover.Status.CURRENT, 'raw_status': discover.Status.CURRENT, 'min_microversion': (1, 3), 'max_microversion': (1, 21), 'next_min_version': None, 'not_before': None}], discover.Discover(self.session, ironic_url).version_data())

    def test_version_data_microversions(self):
        """Validate [min_|max_]version conversion to {min|max}_microversion."""

        def setup_mock(versions_in):
            jsondata = {'versions': [dict({'status': discover.Status.CURRENT, 'id': 'v2.2', 'links': [{'href': V3_URL, 'rel': 'self'}]}, **versions_in)]}
            self.requests_mock.get(V3_URL, status_code=200, json=jsondata)

        def test_ok(versions_in, versions_out):
            setup_mock(versions_in)
            self.assertEqual([dict({'collection': None, 'version': (2, 2), 'url': V3_URL, 'status': discover.Status.CURRENT, 'raw_status': discover.Status.CURRENT}, **versions_out)], discover.Discover(self.session, V3_URL).version_data())

        def test_exc(versions_in):
            setup_mock(versions_in)
            self.assertRaises(TypeError, discover.Discover(self.session, V3_URL).version_data)
        test_ok({}, {'min_microversion': None, 'max_microversion': None, 'next_min_version': None, 'not_before': None})
        test_ok({'version': '2.2'}, {'min_microversion': None, 'max_microversion': (2, 2), 'next_min_version': None, 'not_before': None})
        test_ok({'min_version': '2', 'version': 'foo', 'max_version': '2.2'}, {'min_microversion': (2, 0), 'max_microversion': (2, 2), 'next_min_version': None, 'not_before': None})
        test_ok({'min_version': '', 'version': '2.1', 'max_version': ''}, {'min_microversion': None, 'max_microversion': (2, 1), 'next_min_version': None, 'not_before': None})
        test_ok({'min_version': '2', 'max_version': '2.2', 'next_min_version': '2.1', 'not_before': '2019-07-01'}, {'min_microversion': (2, 0), 'max_microversion': (2, 2), 'next_min_version': (2, 1), 'not_before': '2019-07-01'})
        test_exc({'min_version': 'foo', 'max_version': '2.1'})
        test_exc({'min_version': '2.1', 'max_version': 'foo'})
        test_exc({'min_version': '2.1', 'version': 'foo'})
        test_exc({'next_min_version': 'bogus', 'not_before': '2019-07-01'})

    def test_endpoint_data_noauth_discover(self):
        mock = self.requests_mock.get(BASE_URL, status_code=200, json=V3_VERSION_LIST)
        self.requests_mock.get(V3_URL, status_code=200, json=V3_VERSION_ENTRY)
        plugin = noauth.NoAuth(endpoint=BASE_URL)
        data = plugin.get_endpoint_data(self.session)
        self.assertEqual(data.api_version, (3, 0))
        self.assertEqual(data.url, V3_URL)
        self.assertEqual(plugin.get_api_major_version(self.session), (3, 0))
        self.assertEqual(plugin.get_endpoint(self.session), BASE_URL)
        self.assertTrue(mock.called_once)

    def test_endpoint_data_noauth_versioned_discover(self):
        self.requests_mock.get(BASE_URL, status_code=200, json=V3_VERSION_LIST)
        self.requests_mock.get(V3_URL, status_code=200, json=V3_VERSION_ENTRY)
        plugin = noauth.NoAuth(endpoint=V3_URL)
        data = plugin.get_endpoint_data(self.session)
        self.assertEqual(data.api_version, (3, 0))
        self.assertEqual(data.url, V3_URL)
        self.assertEqual(plugin.get_api_major_version(self.session), (3, 0))
        self.assertEqual(plugin.get_endpoint(self.session), V3_URL)

    def test_endpoint_data_noauth_no_discover(self):
        plugin = noauth.NoAuth(endpoint=V3_URL)
        data = plugin.get_endpoint_data(self.session, discover_versions=False)
        self.assertEqual(data.api_version, (3, 0))
        self.assertEqual(data.url, V3_URL)
        self.assertEqual(plugin.get_api_major_version(self.session), (3, 0))
        self.assertEqual(plugin.get_endpoint(self.session), V3_URL)

    def test_endpoint_data_noauth_override_no_discover(self):
        plugin = noauth.NoAuth()
        data = plugin.get_endpoint_data(self.session, endpoint_override=V3_URL, discover_versions=False)
        self.assertEqual(data.api_version, (3, 0))
        self.assertEqual(data.url, V3_URL)
        self.assertEqual(plugin.get_endpoint(self.session, endpoint_override=V3_URL), V3_URL)

    def test_endpoint_data_http_basic_discover(self):
        self.requests_mock.get(BASE_URL, status_code=200, json=V3_VERSION_LIST)
        self.requests_mock.get(V3_URL, status_code=200, json=V3_VERSION_ENTRY)
        plugin = http_basic.HTTPBasicAuth(endpoint=V3_URL)
        data = plugin.get_endpoint_data(self.session)
        self.assertEqual(data.api_version, (3, 0))
        self.assertEqual(data.url, V3_URL)
        self.assertEqual(plugin.get_api_major_version(self.session), (3, 0))
        self.assertEqual(plugin.get_endpoint(self.session), V3_URL)

    def test_endpoint_data_http_basic_no_discover(self):
        plugin = http_basic.HTTPBasicAuth(endpoint=V3_URL)
        data = plugin.get_endpoint_data(self.session, discover_versions=False)
        self.assertEqual(data.api_version, (3, 0))
        self.assertEqual(data.url, V3_URL)
        self.assertEqual(plugin.get_api_major_version(self.session), (3, 0))
        self.assertEqual(plugin.get_endpoint(self.session), V3_URL)

    def test_endpoint_data_http_basic_override_no_discover(self):
        plugin = http_basic.HTTPBasicAuth()
        data = plugin.get_endpoint_data(self.session, endpoint_override=V3_URL, discover_versions=False)
        self.assertEqual(data.api_version, (3, 0))
        self.assertEqual(data.url, V3_URL)
        self.assertEqual(plugin.get_api_major_version(self.session, endpoint_override=V3_URL), (3, 0))
        self.assertEqual(plugin.get_endpoint(self.session, endpoint_override=V3_URL), V3_URL)

    def test_endpoint_data_noauth_adapter(self):
        self.requests_mock.get(BASE_URL, status_code=200, json=V3_VERSION_LIST)
        self.requests_mock.get(V3_URL, status_code=200, json=V3_VERSION_ENTRY)
        client = adapter.Adapter(session.Session(noauth.NoAuth()), endpoint_override=BASE_URL)
        data = client.get_endpoint_data()
        self.assertEqual(data.api_version, (3, 0))
        self.assertEqual(data.url, V3_URL)
        self.assertEqual(client.get_api_major_version(), (3, 0))
        self.assertEqual(client.get_endpoint(), BASE_URL)

    def test_endpoint_data_noauth_versioned_adapter(self):
        mock = self.requests_mock.get(V3_URL, status_code=200, json=V3_VERSION_ENTRY)
        client = adapter.Adapter(session.Session(noauth.NoAuth()), endpoint_override=V3_URL)
        data = client.get_endpoint_data()
        self.assertEqual(data.api_version, (3, 0))
        self.assertEqual(data.url, V3_URL)
        self.assertEqual(client.get_api_major_version(), (3, 0))
        self.assertEqual(client.get_endpoint(), V3_URL)
        self.assertTrue(mock.called_once)

    def test_endpoint_data_token_endpoint_discover(self):
        mock = self.requests_mock.get(V3_URL, status_code=200, json=V3_VERSION_ENTRY)
        plugin = token_endpoint.Token(endpoint=V3_URL, token='bogus')
        data = plugin.get_endpoint_data(self.session)
        self.assertEqual(data.api_version, (3, 0))
        self.assertEqual(data.url, V3_URL)
        self.assertEqual(plugin.get_api_major_version(self.session), (3, 0))
        self.assertEqual(plugin.get_endpoint(self.session), V3_URL)
        self.assertTrue(mock.called_once)

    def test_endpoint_data_token_endpoint_no_discover(self):
        plugin = token_endpoint.Token(endpoint=V3_URL, token='bogus')
        data = plugin.get_endpoint_data(self.session, discover_versions=False)
        self.assertEqual(data.api_version, (3, 0))
        self.assertEqual(data.url, V3_URL)
        self.assertEqual(plugin.get_api_major_version(self.session), (3, 0))
        self.assertEqual(plugin.get_endpoint(self.session), V3_URL)

    def test_endpoint_data_token_endpoint_adapter(self):
        mock = self.requests_mock.get(V3_URL, status_code=200, json=V3_VERSION_ENTRY)
        plugin = token_endpoint.Token(endpoint=V3_URL, token='bogus')
        client = adapter.Adapter(session.Session(plugin))
        data = client.get_endpoint_data()
        self.assertEqual(data.api_version, (3, 0))
        self.assertEqual(data.url, V3_URL)
        self.assertEqual(client.get_api_major_version(), (3, 0))
        self.assertEqual(client.get_endpoint(), V3_URL)
        self.assertTrue(mock.called_once)

    def test_data_for_url(self):
        mock = self.requests_mock.get(V3_URL, status_code=200, json=V3_VERSION_ENTRY)
        disc = discover.Discover(self.session, V3_URL)
        for url in (V3_URL, V3_URL + '/'):
            data = disc.versioned_data_for(url=url)
            self.assertEqual(data['version'], (3, 0))
            self.assertEqual(data['raw_status'], 'stable')
            self.assertEqual(data['url'], V3_URL)
        self.assertTrue(mock.called_once)

    def test_data_for_no_version(self):
        mock = self.requests_mock.get(V3_URL, status_code=200, json=V3_VERSION_ENTRY)
        disc = discover.Discover(self.session, V3_URL)
        data = disc.versioned_data_for()
        self.assertEqual(data['version'], (3, 0))
        self.assertEqual(data['raw_status'], 'stable')
        self.assertEqual(data['url'], V3_URL)
        self.assertRaises(TypeError, disc.data_for, version=None)
        self.assertTrue(mock.called_once)

    def test_keystone_version_data(self):
        mock = self.requests_mock.get(BASE_URL, status_code=300, json=V3_VERSION_LIST)
        disc = discover.Discover(self.session, BASE_URL)
        raw_data = disc.raw_version_data()
        clean_data = disc.version_data()
        self.assertEqual(2, len(raw_data))
        self.assertEqual(2, len(clean_data))
        for v in raw_data:
            self.assertIn(v['id'], ('v2.0', 'v3.0'))
            self.assertEqual(v['updated'], UPDATED)
            self.assertEqual(v['status'], 'stable')
            if v['id'] == 'v3.0':
                self.assertEqual(v['media-types'], V3_MEDIA_TYPES)
        for v in clean_data:
            self.assertIn(v['version'], ((2, 0), (3, 0)))
            self.assertEqual(v['raw_status'], 'stable')
        valid_v3_versions = (disc.data_for('v3.0'), disc.data_for('3.latest'), disc.data_for('latest'), disc.versioned_data_for(min_version='v3.0', max_version='v3.latest'), disc.versioned_data_for(min_version='3'), disc.versioned_data_for(min_version='3.latest'), disc.versioned_data_for(min_version='latest'), disc.versioned_data_for(min_version='3.latest', max_version='latest'), disc.versioned_data_for(min_version='latest', max_version='latest'), disc.versioned_data_for(min_version=2), disc.versioned_data_for(min_version='2.latest'))
        for version in valid_v3_versions:
            self.assertEqual((3, 0), version['version'])
            self.assertEqual('stable', version['raw_status'])
            self.assertEqual(V3_URL, version['url'])
        valid_v2_versions = (disc.data_for(2), disc.data_for('2.latest'), disc.versioned_data_for(min_version=2, max_version=(2, discover.LATEST)), disc.versioned_data_for(min_version='2.latest', max_version='2.latest'))
        for version in valid_v2_versions:
            self.assertEqual((2, 0), version['version'])
            self.assertEqual('stable', version['raw_status'])
            self.assertEqual(V2_URL, version['url'])
        self.assertIsNone(disc.url_for('v4'))
        self.assertIsNone(disc.versioned_url_for(min_version='v4', max_version='v4.latest'))
        self.assertEqual(V3_URL, disc.url_for('v3'))
        self.assertEqual(V3_URL, disc.versioned_url_for(min_version='v3', max_version='v3.latest'))
        self.assertEqual(V2_URL, disc.url_for('v2'))
        self.assertEqual(V2_URL, disc.versioned_url_for(min_version='v2', max_version='v2.latest'))
        self.assertTrue(mock.called_once)

    def test_cinder_version_data(self):
        mock = self.requests_mock.get(BASE_URL, status_code=300, json=CINDER_EXAMPLES)
        disc = discover.Discover(self.session, BASE_URL)
        raw_data = disc.raw_version_data()
        clean_data = disc.version_data()
        self.assertEqual(3, len(raw_data))
        for v in raw_data:
            self.assertEqual(v['status'], discover.Status.CURRENT)
            if v['id'] == 'v1.0':
                self.assertEqual(v['updated'], '2012-01-04T11:33:21Z')
            elif v['id'] == 'v2.0':
                self.assertEqual(v['updated'], '2012-11-21T11:33:21Z')
            elif v['id'] == 'v3.0':
                self.assertEqual(v['updated'], '2012-11-21T11:33:21Z')
            else:
                self.fail('Invalid version found')
        v1_url = '%sv1/' % BASE_URL
        v2_url = '%sv2/' % BASE_URL
        v3_url = '%sv3/' % BASE_URL
        self.assertEqual(clean_data, [{'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'version': (1, 0), 'url': v1_url, 'status': discover.Status.CURRENT, 'raw_status': discover.Status.CURRENT}, {'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'version': (2, 0), 'url': v2_url, 'status': discover.Status.CURRENT, 'raw_status': discover.Status.CURRENT}, {'collection': BASE_URL, 'max_microversion': (3, 27), 'min_microversion': (3, 0), 'next_min_version': (3, 4), 'not_before': u'2019-12-31', 'version': (3, 0), 'url': v3_url, 'status': discover.Status.CURRENT, 'raw_status': discover.Status.CURRENT}])
        for version in (disc.data_for('v2.0'), disc.versioned_data_for(min_version='v2.0', max_version='v2.latest')):
            self.assertEqual((2, 0), version['version'])
            self.assertEqual(discover.Status.CURRENT, version['raw_status'])
            self.assertEqual(v2_url, version['url'])
        for version in (disc.data_for(1), disc.versioned_data_for(min_version=(1,), max_version=(1, discover.LATEST))):
            self.assertEqual((1, 0), version['version'])
            self.assertEqual(discover.Status.CURRENT, version['raw_status'])
            self.assertEqual(v1_url, version['url'])
        self.assertIsNone(disc.url_for('v4'))
        self.assertIsNone(disc.versioned_url_for(min_version='v4', max_version='v4.latest'))
        self.assertEqual(v3_url, disc.url_for('v3'))
        self.assertEqual(v3_url, disc.versioned_url_for(min_version='v3', max_version='v3.latest'))
        self.assertEqual(v2_url, disc.url_for('v2'))
        self.assertEqual(v2_url, disc.versioned_url_for(min_version='v2', max_version='v2.latest'))
        self.assertEqual(v1_url, disc.url_for('v1'))
        self.assertEqual(v1_url, disc.versioned_url_for(min_version='v1', max_version='v1.latest'))
        self.assertTrue(mock.called_once)

    def test_glance_version_data(self):
        mock = self.requests_mock.get(BASE_URL, status_code=200, json=GLANCE_EXAMPLES)
        disc = discover.Discover(self.session, BASE_URL)
        raw_data = disc.raw_version_data()
        clean_data = disc.version_data()
        self.assertEqual(5, len(raw_data))
        for v in raw_data:
            if v['id'] in ('v2.2', 'v1.1'):
                self.assertEqual(v['status'], discover.Status.CURRENT)
            elif v['id'] in ('v2.1', 'v2.0', 'v1.0'):
                self.assertEqual(v['status'], discover.Status.SUPPORTED)
            else:
                self.fail('Invalid version found')
        v1_url = '%sv1/' % BASE_URL
        v2_url = '%sv2/' % BASE_URL
        self.assertEqual(clean_data, [{'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'version': (1, 0), 'url': v1_url, 'status': discover.Status.SUPPORTED, 'raw_status': discover.Status.SUPPORTED}, {'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'version': (1, 1), 'url': v1_url, 'status': discover.Status.CURRENT, 'raw_status': discover.Status.CURRENT}, {'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'version': (2, 0), 'url': v2_url, 'status': discover.Status.SUPPORTED, 'raw_status': discover.Status.SUPPORTED}, {'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'version': (2, 1), 'url': v2_url, 'status': discover.Status.SUPPORTED, 'raw_status': discover.Status.SUPPORTED}, {'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'version': (2, 2), 'url': v2_url, 'status': discover.Status.CURRENT, 'raw_status': discover.Status.CURRENT}])
        for ver in (2, 2.1, 2.2):
            for version in (disc.data_for(ver), disc.versioned_data_for(min_version=ver, max_version=(2, discover.LATEST))):
                self.assertEqual((2, 2), version['version'])
                self.assertEqual(discover.Status.CURRENT, version['raw_status'])
                self.assertEqual(v2_url, version['url'])
                self.assertEqual(v2_url, disc.url_for(ver))
        for ver in (1, 1.1):
            for version in (disc.data_for(ver), disc.versioned_data_for(min_version=ver, max_version=(1, discover.LATEST))):
                self.assertEqual((1, 1), version['version'])
                self.assertEqual(discover.Status.CURRENT, version['raw_status'])
                self.assertEqual(v1_url, version['url'])
                self.assertEqual(v1_url, disc.url_for(ver))
        self.assertIsNone(disc.url_for('v3'))
        self.assertIsNone(disc.versioned_url_for(min_version='v3', max_version='v3.latest'))
        self.assertIsNone(disc.url_for('v2.3'))
        self.assertIsNone(disc.versioned_url_for(min_version='v2.3', max_version='v2.latest'))
        self.assertTrue(mock.called_once)

    def test_allow_deprecated(self):
        status = 'deprecated'
        version_list = [{'id': 'v3.0', 'links': [{'href': V3_URL, 'rel': 'self'}], 'media-types': V3_MEDIA_TYPES, 'status': status, 'updated': UPDATED}]
        self.requests_mock.get(BASE_URL, json={'versions': version_list})
        disc = discover.Discover(self.session, BASE_URL)
        versions = disc.version_data(allow_deprecated=False)
        self.assertEqual(0, len(versions))
        versions = disc.version_data(allow_deprecated=True)
        self.assertEqual(1, len(versions))
        self.assertEqual(status, versions[0]['raw_status'])
        self.assertEqual(V3_URL, versions[0]['url'])
        self.assertEqual((3, 0), versions[0]['version'])

    def test_allow_experimental(self):
        status = 'experimental'
        version_list = [{'id': 'v3.0', 'links': [{'href': V3_URL, 'rel': 'self'}], 'media-types': V3_MEDIA_TYPES, 'status': status, 'updated': UPDATED}]
        self.requests_mock.get(BASE_URL, json={'versions': version_list})
        disc = discover.Discover(self.session, BASE_URL)
        versions = disc.version_data()
        self.assertEqual(0, len(versions))
        versions = disc.version_data(allow_experimental=True)
        self.assertEqual(1, len(versions))
        self.assertEqual(status, versions[0]['raw_status'])
        self.assertEqual(V3_URL, versions[0]['url'])
        self.assertEqual((3, 0), versions[0]['version'])

    def test_allow_unknown(self):
        status = 'abcdef'
        version_list = fixture.DiscoveryList(BASE_URL, v2=False, v3_status=status)
        self.requests_mock.get(BASE_URL, json=version_list)
        disc = discover.Discover(self.session, BASE_URL)
        versions = disc.version_data()
        self.assertEqual(0, len(versions))
        versions = disc.version_data(allow_unknown=True)
        self.assertEqual(1, len(versions))
        self.assertEqual(status, versions[0]['raw_status'])
        self.assertEqual(V3_URL, versions[0]['url'])
        self.assertEqual((3, 0), versions[0]['version'])

    def test_ignoring_invalid_links(self):
        version_list = [{'id': 'v3.0', 'links': [{'href': V3_URL, 'rel': 'self'}], 'media-types': V3_MEDIA_TYPES, 'status': 'stable', 'updated': UPDATED}, {'id': 'v3.1', 'media-types': V3_MEDIA_TYPES, 'status': 'stable', 'updated': UPDATED}, {'media-types': V3_MEDIA_TYPES, 'status': 'stable', 'updated': UPDATED, 'links': [{'href': V3_URL, 'rel': 'self'}]}]
        self.requests_mock.get(BASE_URL, json={'versions': version_list})
        disc = discover.Discover(self.session, BASE_URL)
        versions = disc.raw_version_data()
        self.assertEqual(3, len(versions))
        versions = disc.version_data()
        self.assertEqual(1, len(versions))