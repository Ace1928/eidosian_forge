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
@unit.skip_if_cache_disabled('resource')
@unit.skip_if_no_multiple_domains_support
def test_project_rename_invalidates_get_project_by_name_cache(self):
    domain = unit.new_domain_ref()
    project = unit.new_project_ref(domain_id=domain['id'])
    project_id = project['id']
    project_name = project['name']
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    PROVIDERS.resource_api.create_project(project_id, project)
    PROVIDERS.resource_api.get_project_by_name(project_name, domain['id'])
    project['name'] = uuid.uuid4().hex
    PROVIDERS.resource_api.update_project(project_id, project)
    self.assertRaises(exception.ProjectNotFound, PROVIDERS.resource_api.get_project_by_name, project_name, domain['id'])