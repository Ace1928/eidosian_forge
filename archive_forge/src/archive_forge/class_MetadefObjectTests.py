import copy
import os
import os.path
from glance.common import config
from glance.common import exception
from glance import context
from glance.db.sqlalchemy import metadata
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
class MetadefObjectTests(object):

    def test_object_create(self):
        fixture = build_namespace_fixture()
        created_ns = self.db_api.metadef_namespace_create(self.context, fixture)
        self.assertIsNotNone(created_ns)
        self._assert_saved_fields(fixture, created_ns)
        fixture_object = build_object_fixture(namespace_id=created_ns['id'])
        created_object = self.db_api.metadef_object_create(self.context, created_ns['namespace'], fixture_object)
        self._assert_saved_fields(fixture_object, created_object)

    def test_object_create_duplicate(self):
        fixture = build_namespace_fixture()
        created_ns = self.db_api.metadef_namespace_create(self.context, fixture)
        self.assertIsNotNone(created_ns)
        self._assert_saved_fields(fixture, created_ns)
        fixture_object = build_object_fixture(namespace_id=created_ns['id'])
        created_object = self.db_api.metadef_object_create(self.context, created_ns['namespace'], fixture_object)
        self._assert_saved_fields(fixture_object, created_object)
        self.assertRaises(exception.Duplicate, self.db_api.metadef_object_create, self.context, created_ns['namespace'], fixture_object)

    def test_object_get(self):
        fixture_ns = build_namespace_fixture()
        created_ns = self.db_api.metadef_namespace_create(self.context, fixture_ns)
        self.assertIsNotNone(created_ns)
        self._assert_saved_fields(fixture_ns, created_ns)
        fixture_object = build_object_fixture(namespace_id=created_ns['id'])
        created_object = self.db_api.metadef_object_create(self.context, created_ns['namespace'], fixture_object)
        found_object = self.db_api.metadef_object_get(self.context, created_ns['namespace'], created_object['name'])
        self._assert_saved_fields(fixture_object, found_object)

    def test_object_get_all(self):
        ns_fixture = build_namespace_fixture()
        ns_created = self.db_api.metadef_namespace_create(self.context, ns_fixture)
        self.assertIsNotNone(ns_created, 'Could not create a namespace.')
        self._assert_saved_fields(ns_fixture, ns_created)
        fixture1 = build_object_fixture(namespace_id=ns_created['id'])
        created_o1 = self.db_api.metadef_object_create(self.context, ns_created['namespace'], fixture1)
        self.assertIsNotNone(created_o1, 'Could not create an object.')
        fixture2 = build_object_fixture(namespace_id=ns_created['id'], name='test-object-2')
        created_o2 = self.db_api.metadef_object_create(self.context, ns_created['namespace'], fixture2)
        self.assertIsNotNone(created_o2, 'Could not create an object.')
        found = self.db_api.metadef_object_get_all(self.context, ns_created['namespace'])
        self.assertEqual(2, len(found))

    def test_object_update(self):
        delta = {'name': 'New-name', 'json_schema': 'new-schema', 'required': 'new-required'}
        fixture_ns = build_namespace_fixture()
        created_ns = self.db_api.metadef_namespace_create(self.context, fixture_ns)
        self.assertIsNotNone(created_ns['namespace'])
        object_fixture = build_object_fixture(namespace_id=created_ns['id'])
        created_object = self.db_api.metadef_object_create(self.context, created_ns['namespace'], object_fixture)
        self.assertIsNotNone(created_object, 'Could not create an object.')
        delta_dict = {}
        delta_dict.update(delta.copy())
        updated = self.db_api.metadef_object_update(self.context, created_ns['namespace'], created_object['id'], delta_dict)
        self.assertEqual(delta['name'], updated['name'])
        self.assertEqual(delta['json_schema'], updated['json_schema'])

    def test_object_delete(self):
        fixture_ns = build_namespace_fixture()
        created_ns = self.db_api.metadef_namespace_create(self.context, fixture_ns)
        self.assertIsNotNone(created_ns['namespace'])
        object_fixture = build_object_fixture(namespace_id=created_ns['id'])
        created_object = self.db_api.metadef_object_create(self.context, created_ns['namespace'], object_fixture)
        self.assertIsNotNone(created_object, 'Could not create an object.')
        self.db_api.metadef_object_delete(self.context, created_ns['namespace'], created_object['name'])
        self.assertRaises(exception.NotFound, self.db_api.metadef_object_get, self.context, created_ns['namespace'], created_object['name'])