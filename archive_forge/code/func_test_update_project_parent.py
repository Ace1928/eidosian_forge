import copy
from unittest import mock
import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
import keystone.conf
from keystone import exception
from keystone.resource.backends import sql as resource_sql
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import utils as test_utils
def test_update_project_parent(self):
    projects_hierarchy = self._create_projects_hierarchy(hierarchy_size=3)
    project1 = projects_hierarchy[0]
    project2 = projects_hierarchy[1]
    project3 = projects_hierarchy[2]
    self.assertEqual(project3.get('parent_id'), project2['id'])
    project3['parent_id'] = project1['id']
    self.assertRaises(exception.ForbiddenNotSecurity, PROVIDERS.resource_api.update_project, project3['id'], project3)