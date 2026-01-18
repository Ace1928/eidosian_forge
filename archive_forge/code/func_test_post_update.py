import uuid
from pycadf import cadftaxonomy as taxonomy
import webob
from keystonemiddleware import audit
from keystonemiddleware.tests.unit.audit import base
def test_post_update(self):
    url = 'http://admin_host:8774/v2/%s/servers/%s' % (uuid.uuid4().hex, uuid.uuid4().hex)
    payload = self.get_payload('POST', url)
    self.assertEqual(payload['target']['typeURI'], 'service/compute/servers/server')
    self.assertEqual(payload['action'], 'update')
    self.assertEqual(payload['outcome'], 'pending')