import uuid
from testtools import matchers
from openstack.tests.unit import base
def _dummy_url(self):
    return 'https://%s.example.com/' % uuid.uuid4().hex