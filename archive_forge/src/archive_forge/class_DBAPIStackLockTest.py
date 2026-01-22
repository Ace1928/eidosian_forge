import datetime
import json
import time
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_db import exception as db_exception
from oslo_utils import timeutils
from sqlalchemy import orm
from sqlalchemy.orm import exc
from sqlalchemy.orm import session
from heat.common import context
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.db import api as db_api
from heat.db import models
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine import resource as rsrc
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.engine import template_files
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
class DBAPIStackLockTest(common.HeatTestCase):

    def setUp(self):
        super(DBAPIStackLockTest, self).setUp()
        self.ctx = utils.dummy_context()
        self.template = create_raw_template(self.ctx)
        self.user_creds = create_user_creds(self.ctx)
        self.stack = create_stack(self.ctx, self.template, self.user_creds)

    def test_stack_lock_create_success(self):
        observed = db_api.stack_lock_create(self.ctx, self.stack.id, UUID1)
        self.assertIsNone(observed)

    def test_stack_lock_create_fail_double_same(self):
        db_api.stack_lock_create(self.ctx, self.stack.id, UUID1)
        observed = db_api.stack_lock_create(self.ctx, self.stack.id, UUID1)
        self.assertEqual(UUID1, observed)

    def test_stack_lock_create_fail_double_different(self):
        db_api.stack_lock_create(self.ctx, self.stack.id, UUID1)
        observed = db_api.stack_lock_create(self.ctx, self.stack.id, UUID2)
        self.assertEqual(UUID1, observed)

    def test_stack_lock_get_id_success(self):
        db_api.stack_lock_create(self.ctx, self.stack.id, UUID1)
        observed = db_api.stack_lock_get_engine_id(self.ctx, self.stack.id)
        self.assertEqual(UUID1, observed)

    def test_stack_lock_get_id_return_none(self):
        observed = db_api.stack_lock_get_engine_id(self.ctx, self.stack.id)
        self.assertIsNone(observed)

    def test_stack_lock_steal_success(self):
        db_api.stack_lock_create(self.ctx, self.stack.id, UUID1)
        observed = db_api.stack_lock_steal(self.ctx, self.stack.id, UUID1, UUID2)
        self.assertIsNone(observed)

    def test_stack_lock_steal_fail_gone(self):
        db_api.stack_lock_create(self.ctx, self.stack.id, UUID1)
        db_api.stack_lock_release(self.ctx, self.stack.id, UUID1)
        observed = db_api.stack_lock_steal(self.ctx, self.stack.id, UUID1, UUID2)
        self.assertTrue(observed)

    def test_stack_lock_steal_fail_stolen(self):
        db_api.stack_lock_create(self.ctx, self.stack.id, UUID1)
        db_api.stack_lock_release(self.ctx, self.stack.id, UUID1)
        db_api.stack_lock_create(self.ctx, self.stack.id, UUID2)
        observed = db_api.stack_lock_steal(self.ctx, self.stack.id, UUID3, UUID2)
        self.assertEqual(UUID2, observed)

    def test_stack_lock_release_success(self):
        db_api.stack_lock_create(self.ctx, self.stack.id, UUID1)
        observed = db_api.stack_lock_release(self.ctx, self.stack.id, UUID1)
        self.assertIsNone(observed)

    def test_stack_lock_release_fail_double(self):
        db_api.stack_lock_create(self.ctx, self.stack.id, UUID1)
        db_api.stack_lock_release(self.ctx, self.stack.id, UUID1)
        observed = db_api.stack_lock_release(self.ctx, self.stack.id, UUID1)
        self.assertTrue(observed)

    def test_stack_lock_release_fail_wrong_engine_id(self):
        db_api.stack_lock_create(self.ctx, self.stack.id, UUID1)
        observed = db_api.stack_lock_release(self.ctx, self.stack.id, UUID2)
        self.assertTrue(observed)

    @mock.patch.object(time, 'sleep')
    def test_stack_lock_retry_on_deadlock(self, sleep):
        with mock.patch('sqlalchemy.orm.Session.add', side_effect=db_exception.DBDeadlock) as mock_add:
            self.assertRaises(db_exception.DBDeadlock, db_api.stack_lock_create, self.ctx, self.stack.id, UUID1)
            self.assertEqual(4, mock_add.call_count)