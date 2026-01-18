import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_service_catalog_endpoint_id(self):
    token = fixture.V2Token()
    token.set_scope()
    endpoint_id = uuid.uuid4().hex
    public_url = uuid.uuid4().hex
    s = token.add_service('compute')
    s.add_endpoint(public=public_url, id=endpoint_id)
    s.add_endpoint(public=uuid.uuid4().hex)
    auth_ref = access.create(body=token)
    urls = auth_ref.service_catalog.get_urls(interface='public')
    self.assertEqual(2, len(urls))
    urls = auth_ref.service_catalog.get_urls(endpoint_id=endpoint_id, interface='public')
    self.assertEqual((public_url,), urls)
    urls = auth_ref.service_catalog.get_urls(endpoint_id=uuid.uuid4().hex, interface='public')
    self.assertEqual(0, len(urls))
    urls = auth_ref.service_catalog.get_urls(endpoint_id=endpoint_id, service_id=uuid.uuid4().hex, interface='public')
    self.assertEqual((public_url,), urls)