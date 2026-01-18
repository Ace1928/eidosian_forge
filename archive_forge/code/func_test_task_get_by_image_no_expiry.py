import copy
import datetime
import functools
from unittest import mock
import uuid
from oslo_db import exception as db_exception
from oslo_db.sqlalchemy import utils as sqlalchemyutils
from sqlalchemy import sql
from glance.common import exception
from glance.common import timeutils
from glance import context
from glance.db.sqlalchemy import api as db_api
from glance.db.sqlalchemy import models
from glance.tests import functional
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
def test_task_get_by_image_no_expiry(self):
    task_id, tasks = self._test_task_get_by_image(expired=None)
    self.assertEqual(1, len(tasks))
    tasks = self.db_api.task_get_all(self.adm_context)
    self.assertEqual(1, len(tasks))
    self.assertEqual(task_id, tasks[0]['id'])
    self.assertFalse(tasks[0]['deleted'])
    self.assertIsNone(tasks[0]['expires_at'])