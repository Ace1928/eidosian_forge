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
def test_task_update(self):
    self.context.project_id = str(uuid.uuid4())
    result = {'foo': 'bar'}
    task_values = build_task_fixture(owner=self.context.owner, result=result)
    task = self.db_api.task_create(self.adm_context, task_values)
    task_id = task['id']
    fixture = {'status': 'processing', 'message': 'This is a error string'}
    self.delay_inaccurate_clock()
    task = self.db_api.task_update(self.adm_context, task_id, fixture)
    self.assertEqual(task_id, task['id'])
    self.assertEqual(self.context.owner, task['owner'])
    self.assertEqual('import', task['type'])
    self.assertEqual('processing', task['status'])
    self.assertEqual({'ping': 'pong'}, task['input'])
    self.assertEqual(result, task['result'])
    self.assertEqual('This is a error string', task['message'])
    self.assertFalse(task['deleted'])
    self.assertIsNone(task['deleted_at'])
    self.assertIsNone(task['expires_at'])
    self.assertEqual(task_values['created_at'], task['created_at'])
    self.assertGreater(task['updated_at'], task['created_at'])