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
def test_update_project_tags_with_too_many_tags(self):
    project, _ = self._create_project_and_tags()
    tags = [uuid.uuid4().hex for i in range(81)]
    tags.append(uuid.uuid4().hex)
    self.put('/projects/%(project_id)s/tags' % {'project_id': project['id']}, body={'tags': tags}, expected_status=http.client.BAD_REQUEST)