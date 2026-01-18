import datetime
import uuid
from oslo_utils import timeutils
from keystoneauth1 import access
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_will_expire_soon(self):
    expires = timeutils.utcnow() + datetime.timedelta(minutes=5)
    token = fixture.V3Token(expires=expires)
    auth_ref = access.create(body=token)
    self.assertFalse(auth_ref.will_expire_soon(stale_duration=120))
    self.assertTrue(auth_ref.will_expire_soon(stale_duration=301))
    self.assertFalse(auth_ref.will_expire_soon())