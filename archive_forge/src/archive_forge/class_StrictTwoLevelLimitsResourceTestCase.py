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
class StrictTwoLevelLimitsResourceTestCase(ResourceTestCase):

    def setUp(self):
        super(StrictTwoLevelLimitsResourceTestCase, self).setUp()

    def config_overrides(self):
        super(StrictTwoLevelLimitsResourceTestCase, self).config_overrides()
        self.config_fixture.config(group='unified_limit', enforcement_model='strict_two_level')

    def _create_projects_hierarchy(self, hierarchy_size=1):
        if hierarchy_size > 1:
            self.skip_test_overrides("Strict two level limit enforcement model doesn't allow theproject tree depth > 2")
        return super(StrictTwoLevelLimitsResourceTestCase, self)._create_projects_hierarchy(hierarchy_size)

    def test_create_hierarchical_project(self):
        projects = self._create_projects_hierarchy()
        new_ref = unit.new_project_ref(domain_id=self.domain_id, parent_id=projects[1]['project']['id'])
        self.post('/projects', body={'project': new_ref}, expected_status=http.client.FORBIDDEN)