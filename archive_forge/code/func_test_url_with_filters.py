import collections
import oslotest.base as base
import osc_placement.resources.common as common
def test_url_with_filters(self):
    base_url = '/resource_providers'
    expected = '/resource_providers?name=test&uuid=123456'
    filters = collections.OrderedDict([('name', 'test'), ('uuid', 123456)])
    actual = common.url_with_filters(base_url, filters)
    self.assertEqual(expected, actual)