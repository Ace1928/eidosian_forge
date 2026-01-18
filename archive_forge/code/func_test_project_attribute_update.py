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
def test_project_attribute_update(self):
    project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
    PROVIDERS.resource_api.create_project(project['id'], project)
    key = 'description'

    def assert_key_equals(value):
        project_ref = PROVIDERS.resource_api.update_project(project['id'], project)
        self.assertEqual(value, project_ref[key])
        project_ref = PROVIDERS.resource_api.get_project(project['id'])
        self.assertEqual(value, project_ref[key])

    def assert_get_key_is(value):
        project_ref = PROVIDERS.resource_api.update_project(project['id'], project)
        self.assertIs(project_ref.get(key), value)
        project_ref = PROVIDERS.resource_api.get_project(project['id'])
        self.assertIs(project_ref.get(key), value)
    value = ''
    project[key] = value
    assert_key_equals(value)
    value = None
    project[key] = value
    assert_get_key_is(value)
    value = None
    project[key] = value
    assert_get_key_is(value)
    value = ''
    project[key] = value
    assert_key_equals(value)
    value = uuid.uuid4().hex
    project[key] = value
    assert_key_equals(value)