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
def test_task_get_all_owned(self):
    then = timeutils.utcnow() + datetime.timedelta(days=365)
    TENANT1 = str(uuid.uuid4())
    ctxt1 = context.RequestContext(is_admin=False, tenant=TENANT1, auth_token='user:%s:user' % TENANT1)
    task_values = {'type': 'import', 'status': 'pending', 'input': '{"loc": "fake"}', 'owner': TENANT1, 'expires_at': then}
    self.db_api.task_create(ctxt1, task_values)
    TENANT2 = str(uuid.uuid4())
    ctxt2 = context.RequestContext(is_admin=False, tenant=TENANT2, auth_token='user:%s:user' % TENANT2)
    task_values = {'type': 'export', 'status': 'pending', 'input': '{"loc": "fake"}', 'owner': TENANT2, 'expires_at': then}
    self.db_api.task_create(ctxt2, task_values)
    tasks = self.db_api.task_get_all(ctxt1)
    task_owners = set([task['owner'] for task in tasks])
    expected = set([TENANT1])
    self.assertEqual(sorted(expected), sorted(task_owners))