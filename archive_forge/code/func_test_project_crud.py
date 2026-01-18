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
@unit.skip_if_no_multiple_domains_support
def test_project_crud(self):
    domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    project = unit.new_project_ref(domain_id=domain['id'])
    PROVIDERS.resource_api.create_project(project['id'], project)
    project_ref = PROVIDERS.resource_api.get_project(project['id'])
    self.assertLessEqual(project.items(), project_ref.items())
    project['name'] = uuid.uuid4().hex
    PROVIDERS.resource_api.update_project(project['id'], project)
    project_ref = PROVIDERS.resource_api.get_project(project['id'])
    self.assertLessEqual(project.items(), project_ref.items())
    PROVIDERS.resource_api.delete_project(project['id'])
    self.assertRaises(exception.ProjectNotFound, PROVIDERS.resource_api.get_project, project['id'])