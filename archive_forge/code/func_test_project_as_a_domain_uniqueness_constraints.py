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
def test_project_as_a_domain_uniqueness_constraints(self):
    """Test project uniqueness for those acting as domains.

        If it is a project acting as a domain, we can't have two or more with
        the same name.

        """
    project = unit.new_project_ref(is_domain=True)
    project = PROVIDERS.resource_api.create_project(project['id'], project)
    project2 = unit.new_project_ref(is_domain=True)
    project2 = PROVIDERS.resource_api.create_project(project2['id'], project2)
    new_project = project.copy()
    new_project['id'] = uuid.uuid4().hex
    self.assertRaises(exception.Conflict, PROVIDERS.resource_api.create_project, new_project['id'], new_project)
    project2['name'] = project['name']
    self.assertRaises(exception.Conflict, PROVIDERS.resource_api.update_project, project2['id'], project2)
    project2['name'] = uuid.uuid4().hex
    PROVIDERS.resource_api.update_project(project2['id'], project2)
    project3 = unit.new_project_ref(domain_id=CONF.identity.default_domain_id, name=project2['name'])
    PROVIDERS.resource_api.create_project(project3['id'], project3)