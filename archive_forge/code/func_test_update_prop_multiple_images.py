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