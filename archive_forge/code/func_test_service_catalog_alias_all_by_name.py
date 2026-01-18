import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_service_catalog_alias_all_by_name(self):
    auth_ref = access.create(auth_token=uuid.uuid4().hex, body=self.AUTH_RESPONSE_BODY)
    sc = auth_ref.service_catalog
    public_ep = sc.get_endpoints(service_name='cinder', interface='public')
    self.assertEqual(public_ep['volumev2'][0]['region'], 'South')
    self.assertEqual(public_ep['volumev2'][0]['url'], 'http://cinder.south.host/cinderapi/public/v2')
    self.assertEqual(public_ep['volumev3'][0]['region'], 'South')
    self.assertEqual(public_ep['volumev3'][0]['url'], 'http://cinder.south.host/cinderapi/public/v3')
    self.assertEqual(public_ep['block-storage'][0]['region'], 'North')
    self.assertEqual(public_ep['block-storage'][0]['url'], 'http://cinder.north.host/cinderapi/public')