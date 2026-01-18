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
def test_project_update_and_project_get_return_same_response(self):
    project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
    PROVIDERS.resource_api.create_project(project['id'], project)
    updated_project = {'enabled': False}
    updated_project_ref = PROVIDERS.resource_api.update_project(project['id'], updated_project)
    updated_project_ref.pop('extra', None)
    self.assertIs(False, updated_project_ref['enabled'])
    project_ref = PROVIDERS.resource_api.get_project(project['id'])
    self.assertDictEqual(updated_project_ref, project_ref)