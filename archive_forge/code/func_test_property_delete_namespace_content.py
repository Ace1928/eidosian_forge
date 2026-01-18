import copy
import os
import os.path
from glance.common import config
from glance.common import exception
from glance import context
from glance.db.sqlalchemy import metadata
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
def test_property_delete_namespace_content(self):
    fixture_ns = build_namespace_fixture()
    created_ns = self.db_api.metadef_namespace_create(self.context, fixture_ns)
    self.assertIsNotNone(created_ns['namespace'])
    prop_fixture = build_property_fixture(namespace_id=created_ns['id'])
    created_prop = self.db_api.metadef_property_create(self.context, created_ns['namespace'], prop_fixture)
    self.assertIsNotNone(created_prop, 'Could not create a property.')
    self.db_api.metadef_property_delete_namespace_content(self.context, created_ns['namespace'])
    self.assertRaises(exception.NotFound, self.db_api.metadef_property_get, self.context, created_ns['namespace'], created_prop['name'])