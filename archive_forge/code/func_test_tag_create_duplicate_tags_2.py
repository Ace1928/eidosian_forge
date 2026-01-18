import copy
import os
import os.path
from glance.common import config
from glance.common import exception
from glance import context
from glance.db.sqlalchemy import metadata
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
def test_tag_create_duplicate_tags_2(self):
    fixture = build_namespace_fixture()
    created_ns = self.db_api.metadef_namespace_create(self.context, fixture)
    self.assertIsNotNone(created_ns)
    self._assert_saved_fields(fixture, created_ns)
    tags = build_tags_fixture(['Tag1', 'Tag2', 'Tag3'])
    self.db_api.metadef_tag_create_tags(self.context, created_ns['namespace'], tags)
    dup_tag = build_tag_fixture(namespace_id=created_ns['id'], name='Tag3')
    self.assertRaises(exception.Duplicate, self.db_api.metadef_tag_create, self.context, created_ns['namespace'], dup_tag)