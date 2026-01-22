import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
@_user_domain.setter
def _user_domain(self, domain):
    self._user['domain'] = domain