import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
@trustor_user_id.setter
def trustor_user_id(self, value):
    trust = self.root.setdefault('OS-TRUST:trust', {})
    trust.setdefault('trustor_user', {})['id'] = value