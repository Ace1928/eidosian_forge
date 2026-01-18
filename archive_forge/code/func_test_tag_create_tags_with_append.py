import copy
import os
import os.path
from glance.common import config
from glance.common import exception
from glance import context
from glance.db.sqlalchemy import metadata
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
def test_tag_create_tags_with_append(self):
    fixture = build_namespace_fixture()
    created_ns = self.db_api.metadef_namespace_create(self.context, fixture)
    self.assertIsNotNone(created_ns)
    self._assert_saved_fields(fixture, created_ns)
    tags = build_tags_fixture(['Tag1', 'Tag2', 'Tag3'])
    created_tags = self.db_api.metadef_tag_create_tags(self.context, created_ns['namespace'], tags)
    actual = set([tag['name'] for tag in created_tags])
    expected = set(['Tag1', 'Tag2', 'Tag3'])
    self.assertEqual(expected, actual)
    new_tags = build_tags_fixture(['Tag4', 'Tag5', 'Tag6'])
    new_created_tags = self.db_api.metadef_tag_create_tags(self.context, created_ns['namespace'], new_tags, can_append=True)
    actual = set([tag['name'] for tag in new_created_tags])
    expected = set(['Tag4', 'Tag5', 'Tag6'])
    self.assertEqual(expected, actual)
    tags = self.db_api.metadef_tag_get_all(self.context, created_ns['namespace'], sort_key='created_at')
    actual = set([tag['name'] for tag in tags])
    expected = set(['Tag1', 'Tag2', 'Tag3', 'Tag4', 'Tag5', 'Tag6'])
    self.assertEqual(expected, actual)