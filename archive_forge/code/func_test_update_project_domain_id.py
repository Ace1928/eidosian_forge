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
def test_update_project_domain_id(self):
    """Call ``PATCH /projects/{project_id}`` with domain_id.

        A projects's `domain_id` is immutable. Ensure that any attempts to
        update the `domain_id` of a project fails.
        """
    project = unit.new_project_ref(domain_id=self.domain['id'])
    project = PROVIDERS.resource_api.create_project(project['id'], project)
    project['domain_id'] = CONF.identity.default_domain_id
    self.patch('/projects/%(project_id)s' % {'project_id': project['id']}, body={'project': project}, expected_status=exception.ValidationError.code)