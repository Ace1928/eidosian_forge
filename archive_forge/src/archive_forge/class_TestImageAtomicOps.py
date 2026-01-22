from oslo_config import cfg
from oslo_db import options
from oslo_utils.fixture import uuidsentinel as uuids
from glance.common import exception
from glance import context as glance_context
import glance.db.sqlalchemy.api
from glance.db.sqlalchemy import models as db_models
from glance.db.sqlalchemy import models_metadef as metadef_models
import glance.tests.functional.db as db_tests
from glance.tests.functional.db import base
from glance.tests.functional.db import base_metadef
class TestImageAtomicOps(base.TestDriver, base.FunctionalInitWrapper):

    def setUp(self):
        db_tests.load(get_db, reset_db)
        super(TestImageAtomicOps, self).setUp()
        self.addCleanup(db_tests.reset)
        self.image = self.db_api.image_create(self.adm_context, {'status': 'active', 'owner': self.adm_context.owner, 'properties': {'speed': '88mph'}})

    @staticmethod
    def _propdict(list_of_props):
        """
        Convert a list of ImageProperty objects to dict, ignoring
        deleted values.
        """
        return {x.name: x.value for x in list_of_props if x.deleted == 0}

    def assertOnlyImageHasProp(self, image_id, name, value):
        images_with_prop = self.db_api.image_get_all(self.adm_context, {'properties': {name: value}})
        self.assertEqual(1, len(images_with_prop))
        self.assertEqual(image_id, images_with_prop[0]['id'])

    def test_update(self):
        """Try to double-create a property atomically.

        This should ensure that a second attempt to create the property
        atomically fails with Duplicate.
        """
        self.db_api.image_set_property_atomic(self.image['id'], 'test_property', 'foo')
        self.assertOnlyImageHasProp(self.image['id'], 'test_property', 'foo')
        self.assertRaises(exception.Duplicate, self.db_api.image_set_property_atomic, self.image['id'], 'test_property', 'bar')
        image = self.db_api.image_get(self.adm_context, self.image['id'])
        self.assertEqual({'speed': '88mph', 'test_property': 'foo'}, self._propdict(image['properties']))
        self.assertOnlyImageHasProp(self.image['id'], 'test_property', 'foo')

    def test_update_drop_update(self):
        """Try to create, delete, re-create property atomically.

        If we fail to undelete and claim the property, this will
        fail as duplicate.
        """
        self.db_api.image_set_property_atomic(self.image['id'], 'test_property', 'foo')
        image = self.db_api.image_get(self.adm_context, self.image['id'])
        self.assertEqual({'speed': '88mph', 'test_property': 'foo'}, self._propdict(image['properties']))
        self.assertOnlyImageHasProp(self.image['id'], 'test_property', 'foo')
        new_props = self._propdict(image['properties'])
        del new_props['test_property']
        self.db_api.image_update(self.adm_context, self.image['id'], values={'properties': new_props}, purge_props=True)
        image = self.db_api.image_get(self.adm_context, self.image['id'])
        self.assertEqual({'speed': '88mph'}, self._propdict(image['properties']))
        self.db_api.image_set_property_atomic(self.image['id'], 'test_property', 'bar')
        image = self.db_api.image_get(self.adm_context, self.image['id'])
        self.assertEqual({'speed': '88mph', 'test_property': 'bar'}, self._propdict(image['properties']))
        self.assertOnlyImageHasProp(self.image['id'], 'test_property', 'bar')

    def test_update_prop_multiple_images(self):
        """Create and delete properties on two images, then set on one.

        This tests that the resurrect-from-deleted mode of the method only
        matches deleted properties from our image.
        """
        images = self.db_api.image_get_all(self.adm_context)
        image_id1 = images[0]['id']
        image_id2 = images[-1]['id']
        self.db_api.image_set_property_atomic(image_id1, 'test_property', 'foo')
        self.db_api.image_set_property_atomic(image_id2, 'test_property', 'bar')
        self.assertOnlyImageHasProp(image_id1, 'test_property', 'foo')
        self.assertOnlyImageHasProp(image_id2, 'test_property', 'bar')
        self.db_api.image_update(self.adm_context, image_id1, {'properties': {}}, purge_props=True)
        self.db_api.image_update(self.adm_context, image_id2, {'properties': {}}, purge_props=True)
        self.db_api.image_set_property_atomic(image_id2, 'test_property', 'baz')
        self.assertOnlyImageHasProp(image_id2, 'test_property', 'baz')

    def test_delete(self):
        """Try to double-delete a property atomically.

        This should ensure that a second attempt fails.
        """
        self.db_api.image_delete_property_atomic(self.image['id'], 'speed', '88mph')
        self.assertRaises(exception.NotFound, self.db_api.image_delete_property_atomic, self.image['id'], 'speed', '88mph')

    def test_delete_create_delete(self):
        """Try to delete, re-create, and then re-delete property."""
        self.db_api.image_delete_property_atomic(self.image['id'], 'speed', '88mph')
        self.db_api.image_update(self.adm_context, self.image['id'], {'properties': {'speed': '89mph'}}, purge_props=True)
        self.assertRaises(exception.NotFound, self.db_api.image_delete_property_atomic, self.image['id'], 'speed', '88mph')
        self.db_api.image_delete_property_atomic(self.image['id'], 'speed', '89mph')

    def test_image_update_ignores_atomics(self):
        image = self.db_api.image_get_all(self.adm_context)[0]
        self.db_api.image_set_property_atomic(image['id'], 'test1', 'foo')
        self.db_api.image_set_property_atomic(image['id'], 'test2', 'bar')
        self.db_api.image_update(self.adm_context, image['id'], {'properties': {'test1': 'baz', 'test3': 'bat', 'test4': 'yep'}}, purge_props=True, atomic_props=['test1', 'test2', 'test3'])
        image = self.db_api.image_get(self.adm_context, image['id'])
        self.assertEqual({'test1': 'foo', 'test2': 'bar', 'test4': 'yep'}, self._propdict(image['properties']))