import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_service_catalog_url_for_region_names(self):
    auth_ref = access.create(body=self.AUTH_RESPONSE_BODY)
    sc = auth_ref.service_catalog
    url = sc.url_for(service_type='image', region_name='North')
    self.assertEqual(url, 'https://image.north.host/v1/')
    url = sc.url_for(service_type='image', region_name='South')
    self.assertEqual(url, 'https://image.south.host/v1/')
    self.assertRaises(exceptions.EndpointNotFound, sc.url_for, service_type='image', region_name='West')