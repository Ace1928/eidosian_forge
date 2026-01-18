import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
@token_id.setter
def token_id(self, value):
    self._token['id'] = value