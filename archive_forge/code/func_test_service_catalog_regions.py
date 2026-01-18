import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_service_catalog_regions(self):
    self.AUTH_RESPONSE_BODY['token']['region_name'] = 'North'
    auth_ref = access.create(auth_token=uuid.uuid4().hex, body=self.AUTH_RESPONSE_BODY)
    sc = auth_ref.service_catalog
    url = sc.url_for(service_type='image', interface='public')
    self.assertEqual(url, 'http://glance.north.host/glanceapi/public')
    self.AUTH_RESPONSE_BODY['token']['region_name'] = 'South'
    auth_ref = access.create(auth_token=uuid.uuid4().hex, body=self.AUTH_RESPONSE_BODY)
    sc = auth_ref.service_catalog
    url = sc.url_for(service_type='image', region_name='South', interface='internal')
    self.assertEqual(url, 'http://glance.south.host/glanceapi/internal')