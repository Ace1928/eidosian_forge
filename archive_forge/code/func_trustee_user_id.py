import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
@trustee_user_id.setter
def trustee_user_id(self, value):
    self.root.setdefault('trust', {})['trustee_user_id'] = value