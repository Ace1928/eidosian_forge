import testtools
from testtools import matchers
from zunclient.tests.unit import utils
from zunclient.v1 import versions
def _test_version_list(self, endpoint):
    api = utils.FakeAPI(fake_responses, endpoint=endpoint)
    mgr = versions.VersionManager(api)
    api_versions = mgr.list()
    expect = [('GET', 'https://example.com', {}, None)]
    self.assertEqual(expect, api.calls)
    self.assertThat(api_versions, matchers.HasLength(1))