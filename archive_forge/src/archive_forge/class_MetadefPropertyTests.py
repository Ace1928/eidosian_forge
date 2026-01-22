import copy
import os
import os.path
from glance.common import config
from glance.common import exception
from glance import context
from glance.db.sqlalchemy import metadata
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
class MetadefPropertyTests(object):

    def test_property_create(self):
        fixture = build_namespace_fixture()
        created_ns = self.db_api.metadef_namespace_create(self.context, fixture)
        self.assertIsNotNone(created_ns)
        self._assert_saved_fields(fixture, created_ns)
        fixture_prop = build_property_fixture(namespace_id=created_ns['id'])
        created_prop = self.db_api.metadef_property_create(self.context, created_ns['namespace'], fixture_prop)
        self._assert_saved_fields(fixture_prop, created_prop)

    def test_property_create_duplicate(self):
        fixture = build_namespace_fixture()
        created_ns = self.db_api.metadef_namespace_create(self.context, fixture)
        self.assertIsNotNone(created_ns)
        self._assert_saved_fields(fixture, created_ns)
        fixture_prop = build_property_fixture(namespace_id=created_ns['id'])
        created_prop = self.db_api.metadef_property_create(self.context, created_ns['namespace'], fixture_prop)
        self._assert_saved_fields(fixture_prop, created_prop)
        self.assertRaises(exception.Duplicate, self.db_api.metadef_property_create, self.context, created_ns['namespace'], fixture_prop)

    def test_property_get(self):
        fixture_ns = build_namespace_fixture()
        created_ns = self.db_api.metadef_namespace_create(self.context, fixture_ns)
        self.assertIsNotNone(created_ns)
        self._assert_saved_fields(fixture_ns, created_ns)
        fixture_prop = build_property_fixture(namespace_id=created_ns['id'])
        created_prop = self.db_api.metadef_property_create(self.context, created_ns['namespace'], fixture_prop)
        found_prop = self.db_api.metadef_property_get(self.context, created_ns['namespace'], created_prop['name'])
        self._assert_saved_fields(fixture_prop, found_prop)

    def test_property_get_all(self):
        ns_fixture = build_namespace_fixture()
        ns_created = self.db_api.metadef_namespace_create(self.context, ns_fixture)
        self.assertIsNotNone(ns_created, 'Could not create a namespace.')
        self._assert_saved_fields(ns_fixture, ns_created)
        fixture1 = build_property_fixture(namespace_id=ns_created['id'])
        created_p1 = self.db_api.metadef_property_create(self.context, ns_created['namespace'], fixture1)
        self.assertIsNotNone(created_p1, 'Could not create a property.')
        fixture2 = build_property_fixture(namespace_id=ns_created['id'], name='test-prop-2')
        created_p2 = self.db_api.metadef_property_create(self.context, ns_created['namespace'], fixture2)
        self.assertIsNotNone(created_p2, 'Could not create a property.')
        found = self.db_api.metadef_property_get_all(self.context, ns_created['namespace'])
        self.assertEqual(2, len(found))

    def test_property_update(self):
        delta = {'name': 'New-name', 'json_schema': 'new-schema'}
        fixture_ns = build_namespace_fixture()
        created_ns = self.db_api.metadef_namespace_create(self.context, fixture_ns)
        self.assertIsNotNone(created_ns['namespace'])
        prop_fixture = build_property_fixture(namespace_id=created_ns['id'])
        created_prop = self.db_api.metadef_property_create(self.context, created_ns['namespace'], prop_fixture)
        self.assertIsNotNone(created_prop, 'Could not create a property.')
        delta_dict = copy.deepcopy(created_prop)
        delta_dict.update(delta.copy())
        updated = self.db_api.metadef_property_update(self.context, created_ns['namespace'], created_prop['id'], delta_dict)
        self.assertEqual(delta['name'], updated['name'])
        self.assertEqual(delta['json_schema'], updated['json_schema'])

    def test_property_delete(self):
        fixture_ns = build_namespace_fixture()
        created_ns = self.db_api.metadef_namespace_create(self.context, fixture_ns)
        self.assertIsNotNone(created_ns['namespace'])
        prop_fixture = build_property_fixture(namespace_id=created_ns['id'])
        created_prop = self.db_api.metadef_property_create(self.context, created_ns['namespace'], prop_fixture)
        self.assertIsNotNone(created_prop, 'Could not create a property.')
        self.db_api.metadef_property_delete(self.context, created_ns['namespace'], created_prop['name'])
        self.assertRaises(exception.NotFound, self.db_api.metadef_property_get, self.context, created_ns['namespace'], created_prop['name'])

    def test_property_delete_namespace_content(self):
        fixture_ns = build_namespace_fixture()
        created_ns = self.db_api.metadef_namespace_create(self.context, fixture_ns)
        self.assertIsNotNone(created_ns['namespace'])
        prop_fixture = build_property_fixture(namespace_id=created_ns['id'])
        created_prop = self.db_api.metadef_property_create(self.context, created_ns['namespace'], prop_fixture)
        self.assertIsNotNone(created_prop, 'Could not create a property.')
        self.db_api.metadef_property_delete_namespace_content(self.context, created_ns['namespace'])
        self.assertRaises(exception.NotFound, self.db_api.metadef_property_get, self.context, created_ns['namespace'], created_prop['name'])