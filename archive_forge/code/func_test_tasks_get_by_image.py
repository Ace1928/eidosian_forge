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
def test_tasks_get_by_image(self):
    tasks = self.db.tasks_get_by_image(self.context, UUID1)
    self.assertEqual(2, len(tasks))
    for task in tasks:
        self.assertEqual(USER1, task['user_id'])
        self.assertEqual('fake-request-id', task['request_id'])
        self.assertEqual(UUID1, task['image_id'])