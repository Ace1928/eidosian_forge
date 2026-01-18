import uuid
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_trust_scoped(self):
    trust_id = uuid.uuid4().hex
    trustee_user_id = uuid.uuid4().hex
    trustor_user_id = uuid.uuid4().hex
    impersonation = True
    token = fixture.V3Token(trust_id=trust_id, trustee_user_id=trustee_user_id, trustor_user_id=trustor_user_id, trust_impersonation=impersonation)
    trust = token['token']['OS-TRUST:trust']
    self.assertEqual(trust_id, token.trust_id)
    self.assertEqual(trust_id, trust['id'])
    self.assertEqual(trustee_user_id, token.trustee_user_id)
    self.assertEqual(trustee_user_id, trust['trustee_user']['id'])
    self.assertEqual(trustor_user_id, token.trustor_user_id)
    self.assertEqual(trustor_user_id, trust['trustor_user']['id'])
    self.assertEqual(impersonation, token.trust_impersonation)
    self.assertEqual(impersonation, trust['impersonation'])