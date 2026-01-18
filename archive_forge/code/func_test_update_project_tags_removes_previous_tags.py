import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
from keystone.tests.unit import utils as test_utils
def test_update_project_tags_removes_previous_tags(self):
    tag = uuid.uuid4().hex
    project, tags = self._create_project_and_tags(num_of_tags=5)
    self.put('/projects/%(project_id)s/tags/%(value)s' % {'project_id': project['id'], 'value': tag}, expected_status=http.client.CREATED)
    resp = self.put('/projects/%(project_id)s/tags' % {'project_id': project['id']}, body={'tags': tags}, expected_status=http.client.OK)
    self.assertNotIn(tag, resp.result['tags'])
    self.assertIn(tags[1], resp.result['tags'])