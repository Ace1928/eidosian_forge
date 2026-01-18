import uuid
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_unscoped(self):
    user_id = uuid.uuid4().hex
    user_name = uuid.uuid4().hex
    user_domain_id = uuid.uuid4().hex
    user_domain_name = uuid.uuid4().hex
    token = fixture.V3Token(user_id=user_id, user_name=user_name, user_domain_id=user_domain_id, user_domain_name=user_domain_name)
    self.assertEqual(user_id, token.user_id)
    self.assertEqual(user_id, token['token']['user']['id'])
    self.assertEqual(user_name, token.user_name)
    self.assertEqual(user_name, token['token']['user']['name'])
    user_domain = token['token']['user']['domain']
    self.assertEqual(user_domain_id, token.user_domain_id)
    self.assertEqual(user_domain_id, user_domain['id'])
    self.assertEqual(user_domain_name, token.user_domain_name)
    self.assertEqual(user_domain_name, user_domain['name'])