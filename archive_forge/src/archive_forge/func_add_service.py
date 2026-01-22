import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
def add_service(self, type, name=None):
    name = name or uuid.uuid4().hex
    service = _Service(name=name, type=type)
    self.root.setdefault('serviceCatalog', []).append(service)
    return service