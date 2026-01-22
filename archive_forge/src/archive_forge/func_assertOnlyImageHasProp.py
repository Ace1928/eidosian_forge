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
def assertOnlyImageHasProp(self, image_id, name, value):
    images_with_prop = self.db_api.image_get_all(self.adm_context, {'properties': {name: value}})
    self.assertEqual(1, len(images_with_prop))
    self.assertEqual(image_id, images_with_prop[0]['id'])