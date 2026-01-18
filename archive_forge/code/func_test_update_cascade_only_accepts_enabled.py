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
def test_update_cascade_only_accepts_enabled(self):
    new_project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
    PROVIDERS.resource_api.create_project(new_project['id'], new_project)
    new_project['name'] = 'project1'
    self.assertRaises(exception.ValidationError, PROVIDERS.resource_api.update_project, new_project['id'], new_project, cascade=True)