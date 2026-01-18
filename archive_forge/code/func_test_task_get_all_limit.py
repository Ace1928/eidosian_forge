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
def test_task_get_all_limit(self):
    for fixture in self.fixtures:
        self.db_api.task_create(self.adm_context, build_task_fixture(**fixture))
    tasks = self.db_api.task_get_all(self.adm_context, limit=2)
    self.assertEqual(2, len(tasks))
    tasks = self.db_api.task_get_all(self.adm_context, limit=None)
    self.assertEqual(3, len(tasks))
    tasks = self.db_api.task_get_all(self.adm_context, limit=0)
    self.assertEqual(0, len(tasks))