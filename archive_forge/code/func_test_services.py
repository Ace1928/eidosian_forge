import uuid
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_services(self):
    service_type = uuid.uuid4().hex
    service_name = uuid.uuid4().hex
    endpoint_id = uuid.uuid4().hex
    region = uuid.uuid4().hex
    public = uuid.uuid4().hex
    admin = uuid.uuid4().hex
    internal = uuid.uuid4().hex
    token = fixture.V2Token()
    svc = token.add_service(type=service_type, name=service_name)
    svc.add_endpoint(public=public, admin=admin, internal=internal, region=region, id=endpoint_id)
    self.assertEqual(1, len(token['access']['serviceCatalog']))
    service = token['access']['serviceCatalog'][0]['endpoints'][0]
    self.assertEqual(public, service['publicURL'])
    self.assertEqual(internal, service['internalURL'])
    self.assertEqual(admin, service['adminURL'])
    self.assertEqual(region, service['region'])
    self.assertEqual(endpoint_id, service['id'])
    token.remove_service(type=service_type)
    self.assertEqual(0, len(token['access']['serviceCatalog']))