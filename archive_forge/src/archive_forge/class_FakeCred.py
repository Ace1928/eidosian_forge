from keystoneauth1 import session
from heat.common import context
class FakeCred(object):
    id = self.credential_id
    access = self.access
    secret = self.secret