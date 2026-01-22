import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
def add_role(self, name=None, id=None):
    id = id or uuid.uuid4().hex
    name = name or uuid.uuid4().hex
    roles = self._user.setdefault('roles', [])
    roles.append({'name': name})
    self._metadata.setdefault('roles', []).append(id)
    return {'id': id, 'name': name}