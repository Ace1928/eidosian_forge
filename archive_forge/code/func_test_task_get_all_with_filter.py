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
def test_task_get_all_with_filter(self):
    for fixture in self.fixtures:
        self.db_api.task_create(self.adm_context, build_task_fixture(**fixture))
    import_tasks = self.db_api.task_get_all(self.adm_context, filters={'type': 'import'})
    self.assertTrue(import_tasks)
    self.assertEqual(2, len(import_tasks))
    for task in import_tasks:
        self.assertEqual('import', task['type'])
        self.assertEqual(self.context.owner, task['owner'])