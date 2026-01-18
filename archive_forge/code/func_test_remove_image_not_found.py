import datetime
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_db import exception as db_exc
from oslo_utils import encodeutils
from oslo_utils.fixture import uuidsentinel as uuids
from oslo_utils import timeutils
from sqlalchemy import orm as sa_orm
from glance.common import crypt
from glance.common import exception
import glance.context
import glance.db
from glance.db.sqlalchemy import api
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_remove_image_not_found(self):
    fake_uuid = str(uuid.uuid4())
    image = self.image_repo.get(UUID1)
    image.image_id = fake_uuid
    exc = self.assertRaises(exception.ImageNotFound, self.image_repo.remove, image)
    self.assertIn(fake_uuid, encodeutils.exception_to_unicode(exc))