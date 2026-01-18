import copy
import os
import os.path
from glance.common import config
from glance.common import exception
from glance import context
from glance.db.sqlalchemy import metadata
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
def test_object_get(self):
    fixture_ns = build_namespace_fixture()
    created_ns = self.db_api.metadef_namespace_create(self.context, fixture_ns)
    self.assertIsNotNone(created_ns)
    self._assert_saved_fields(fixture_ns, created_ns)
    fixture_object = build_object_fixture(namespace_id=created_ns['id'])
    created_object = self.db_api.metadef_object_create(self.context, created_ns['namespace'], fixture_object)
    found_object = self.db_api.metadef_object_get(self.context, created_ns['namespace'], created_object['name'])
    self._assert_saved_fields(fixture_object, found_object)