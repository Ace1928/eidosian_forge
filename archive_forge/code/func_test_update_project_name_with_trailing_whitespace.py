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
def test_update_project_name_with_trailing_whitespace(self):
    project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
    project_id = project['id']
    project_create = PROVIDERS.resource_api.create_project(project_id, project)
    self.assertEqual(project_id, project_create['id'])
    project_name = project['name'] = project['name'] + '    '
    project_update = PROVIDERS.resource_api.update_project(project_id, project)
    self.assertEqual(project_id, project_update['id'])
    self.assertEqual(project_name.strip(), project_update['name'])