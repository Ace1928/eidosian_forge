import copy
import functools
import random
import http.client
from oslo_serialization import jsonutils
from testtools import matchers as tt_matchers
import webob
from keystone.api import discovery
from keystone.common import json_home
from keystone.tests import unit
class VersionSingleAppTestCase(unit.TestCase):
    """Test running with a single application loaded.

    These are important because when Keystone is running in Apache httpd
    there's only one application loaded for each instance.

    """

    def setUp(self):
        super(VersionSingleAppTestCase, self).setUp()
        self.load_backends()
        self.public_port = random.randint(40000, 60000)
        self.config_fixture.config(public_endpoint='http://localhost:%d' % self.public_port)

    def config_overrides(self):
        super(VersionSingleAppTestCase, self).config_overrides()

    def _paste_in_port(self, response, port):
        for link in response['links']:
            if link['rel'] == 'self':
                link['href'] = port

    def _test_version(self, app_name):
        app = self.loadapp(app_name)
        client = TestClient(app)
        resp = client.get('/')
        self.assertEqual(300, resp.status_int)
        data = jsonutils.loads(resp.body)
        expected = VERSIONS_RESPONSE
        url_with_port = 'http://localhost:%s/v3/' % self.public_port
        for version in expected['versions']['values']:
            if version['id'].startswith('v3'):
                self._paste_in_port(version, url_with_port)
        self.assertIn('Location', resp.headers)
        self.assertEqual(url_with_port, resp.headers['Location'])
        self.assertThat(data, _VersionsEqual(expected))

    def test_public(self):
        self._test_version('public')

    def test_admin(self):
        self._test_version('admin')