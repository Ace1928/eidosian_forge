import copy
import os
import os.path
from glance.common import config
from glance.common import exception
from glance import context
from glance.db.sqlalchemy import metadata
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
class MetadefNamespaceTests(object):

    def test_namespace_create(self):
        fixture = build_namespace_fixture()
        created = self.db_api.metadef_namespace_create(self.context, fixture)
        self.assertIsNotNone(created)
        self._assert_saved_fields(fixture, created)

    def test_namespace_create_duplicate(self):
        fixture = build_namespace_fixture()
        created = self.db_api.metadef_namespace_create(self.context, fixture)
        self.assertIsNotNone(created)
        self._assert_saved_fields(fixture, created)
        self.assertRaises(exception.Duplicate, self.db_api.metadef_namespace_create, self.context, fixture)

    def test_namespace_get(self):
        fixture = build_namespace_fixture()
        created = self.db_api.metadef_namespace_create(self.context, fixture)
        self.assertIsNotNone(created)
        self._assert_saved_fields(fixture, created)
        found = self.db_api.metadef_namespace_get(self.context, created['namespace'])
        self.assertIsNotNone(found, 'Namespace not found.')

    def test_namespace_get_all_with_resource_types_filter(self):
        ns_fixture = build_namespace_fixture()
        ns_created = self.db_api.metadef_namespace_create(self.context, ns_fixture)
        self.assertIsNotNone(ns_created, 'Could not create a namespace.')
        self._assert_saved_fields(ns_fixture, ns_created)
        fixture = build_association_fixture()
        created = self.db_api.metadef_resource_type_association_create(self.context, ns_created['namespace'], fixture)
        self.assertIsNotNone(created, 'Could not create an association.')
        rt_filters = {'resource_types': fixture['name']}
        found = self.db_api.metadef_namespace_get_all(self.context, filters=rt_filters, sort_key='created_at')
        self.assertEqual(1, len(found))
        for item in found:
            self._assert_saved_fields(ns_fixture, item)

    def test_namespace_update(self):
        delta = {'owner': 'New Owner'}
        fixture = build_namespace_fixture()
        created = self.db_api.metadef_namespace_create(self.context, fixture)
        self.assertIsNotNone(created['namespace'])
        self.assertEqual(fixture['namespace'], created['namespace'])
        delta_dict = copy.deepcopy(created)
        delta_dict.update(delta.copy())
        updated = self.db_api.metadef_namespace_update(self.context, created['id'], delta_dict)
        self.assertEqual(delta['owner'], updated['owner'])

    def test_namespace_delete(self):
        fixture = build_namespace_fixture()
        created = self.db_api.metadef_namespace_create(self.context, fixture)
        self.assertIsNotNone(created, 'Could not create a Namespace.')
        self.db_api.metadef_namespace_delete(self.context, created['namespace'])
        self.assertRaises(exception.NotFound, self.db_api.metadef_namespace_get, self.context, created['namespace'])

    def test_namespace_delete_with_content(self):
        fixture_ns = build_namespace_fixture()
        created_ns = self.db_api.metadef_namespace_create(self.context, fixture_ns)
        self._assert_saved_fields(fixture_ns, created_ns)
        fixture_obj = build_object_fixture()
        created_obj = self.db_api.metadef_object_create(self.context, created_ns['namespace'], fixture_obj)
        self.assertIsNotNone(created_obj)
        fixture_prop = build_property_fixture(namespace_id=created_ns['id'])
        created_prop = self.db_api.metadef_property_create(self.context, created_ns['namespace'], fixture_prop)
        self.assertIsNotNone(created_prop)
        fixture_assn = build_association_fixture()
        created_assn = self.db_api.metadef_resource_type_association_create(self.context, created_ns['namespace'], fixture_assn)
        self.assertIsNotNone(created_assn)
        deleted_ns = self.db_api.metadef_namespace_delete(self.context, created_ns['namespace'])
        self.assertRaises(exception.NotFound, self.db_api.metadef_namespace_get, self.context, deleted_ns['namespace'])