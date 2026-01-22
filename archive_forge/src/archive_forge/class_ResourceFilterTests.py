import ddt
from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
@ddt.ddt
class ResourceFilterTests(utils.TestCase):

    @ddt.data({'resource': None, 'query_url': None}, {'resource': 'volume', 'query_url': '?resource=volume'}, {'resource': 'group', 'query_url': '?resource=group'})
    @ddt.unpack
    def test_list_resource_filters(self, resource, query_url):
        cs.resource_filters.list(resource)
        url = '/resource_filters'
        if resource is not None:
            url += query_url
        cs.assert_called('GET', url)