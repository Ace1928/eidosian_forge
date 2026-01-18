import uuid
from openstack.network.v2 import agent
from openstack.tests.functional import base
def validate_uuid(self, s):
    try:
        uuid.UUID(s)
    except Exception:
        return False
    return True