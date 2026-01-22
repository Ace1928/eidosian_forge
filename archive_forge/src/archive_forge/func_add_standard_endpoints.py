import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
def add_standard_endpoints(self, public=None, admin=None, internal=None, region=None):
    ret = []
    if public:
        ret.append(self.add_endpoint('public', public, region=region))
    if admin:
        ret.append(self.add_endpoint('admin', admin, region=region))
    if internal:
        ret.append(self.add_endpoint('internal', internal, region=region))
    return ret