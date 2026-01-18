import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_service_catalog_multiple_service_types(self):
    token = fixture.V2Token()
    token.set_scope()
    for i in range(3):
        s = token.add_service('compute')
        s.add_endpoint(public='public-%d' % i, admin='admin-%d' % i, internal='internal-%d' % i, region='region-%d' % i)
    auth_ref = access.create(body=token)
    urls = auth_ref.service_catalog.get_urls(service_type='compute', interface='publicURL')
    self.assertEqual(set(['public-0', 'public-1', 'public-2']), set(urls))
    urls = auth_ref.service_catalog.get_urls(service_type='compute', interface='publicURL', region_name='region-1')
    self.assertEqual(('public-1',), urls)