import testtools
from testtools import matchers
from zunclient.tests.unit import utils
from zunclient.v1 import services
class ServiceManagerTest(testtools.TestCase):

    def setUp(self):
        super(ServiceManagerTest, self).setUp()
        self.api = utils.FakeAPI(fake_responses)
        self.mgr = services.ServiceManager(self.api)

    def test_service_list(self):
        services = self.mgr.list()
        expect = [('GET', '/v1/services', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertThat(services, matchers.HasLength(2))

    def _test_service_list_with_filters(self, limit=None, marker=None, sort_key=None, sort_dir=None, expect=[]):
        services_filter = self.mgr.list(limit=limit, marker=marker, sort_key=sort_key, sort_dir=sort_dir)
        self.assertEqual(expect, self.api.calls)
        self.assertThat(services_filter, matchers.HasLength(2))

    def test_service_list_with_limit(self):
        expect = [('GET', '/v1/services/?limit=2', {}, None)]
        self._test_service_list_with_filters(limit=2, expect=expect)

    def test_service_list_with_marker(self):
        expect = [('GET', '/v1/services/?marker=%s' % SERVICE2['id'], {}, None)]
        self._test_service_list_with_filters(marker=SERVICE2['id'], expect=expect)

    def test_service_list_with_marker_limit(self):
        expect = [('GET', '/v1/services/?limit=2&marker=%s' % SERVICE2['id'], {}, None)]
        self._test_service_list_with_filters(limit=2, marker=SERVICE2['id'], expect=expect)

    def test_service_list_with_sort_dir(self):
        expect = [('GET', '/v1/services/?sort_dir=asc', {}, None)]
        self._test_service_list_with_filters(sort_dir='asc', expect=expect)

    def test_service_list_with_sort_key(self):
        expect = [('GET', '/v1/services/?sort_key=id', {}, None)]
        self._test_service_list_with_filters(sort_key='id', expect=expect)

    def test_service_list_with_sort_key_dir(self):
        expect = [('GET', '/v1/services/?sort_key=id&sort_dir=desc', {}, None)]
        self._test_service_list_with_filters(sort_key='id', sort_dir='desc', expect=expect)