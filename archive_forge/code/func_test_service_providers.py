import uuid
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_service_providers(self):

    def new_sp():
        return {'id': uuid.uuid4().hex, 'sp_url': uuid.uuid4().hex, 'auth_url': uuid.uuid4().hex}
    ref_service_providers = [new_sp(), new_sp()]
    token = fixture.V3Token()
    for sp in ref_service_providers:
        token.add_service_provider(sp['id'], sp['auth_url'], sp['sp_url'])
    self.assertEqual(ref_service_providers, token.service_providers)
    self.assertEqual(ref_service_providers, token['token']['service_providers'])