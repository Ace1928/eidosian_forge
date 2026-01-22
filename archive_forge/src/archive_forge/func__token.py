import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
@property
def _token(self):
    return self.root.setdefault('token', {})