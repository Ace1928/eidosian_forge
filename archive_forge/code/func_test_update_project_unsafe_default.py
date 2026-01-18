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
def test_update_project_unsafe_default(self):
    """Check default for unsafe names for ``POST /projects``."""
    unsafe_name = 'i am not / safe'
    ref = unit.new_project_ref(name=unsafe_name, domain_id=self.domain_id, parent_id=self.project['parent_id'])
    del ref['id']
    self.patch('/projects/%(project_id)s' % {'project_id': self.project_id}, body={'project': ref})