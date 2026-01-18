import datetime
import uuid
from oslo_utils import timeutils
from keystoneauth1 import access
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_trusts(self):
    user_id = uuid.uuid4().hex
    trust_id = uuid.uuid4().hex
    token = fixture.V2Token(user_id=user_id, trust_id=trust_id)
    token.set_scope()
    token.add_role()
    auth_ref = access.create(body=token)
    self.assertIsInstance(auth_ref, access.AccessInfoV2)
    self.assertEqual(trust_id, auth_ref.trust_id)
    self.assertEqual(user_id, auth_ref.trustee_user_id)
    self.assertEqual(trust_id, token['access']['trust']['id'])