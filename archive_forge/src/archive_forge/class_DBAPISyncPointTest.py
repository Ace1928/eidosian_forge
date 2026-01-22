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
class DBAPISyncPointTest(common.HeatTestCase):

    def setUp(self):
        super(DBAPISyncPointTest, self).setUp()
        self.ctx = utils.dummy_context()
        self.template = create_raw_template(self.ctx)
        self.user_creds = create_user_creds(self.ctx)
        self.stack = create_stack(self.ctx, self.template, self.user_creds)
        self.resources = [create_resource(self.ctx, self.stack, name='res1'), create_resource(self.ctx, self.stack, name='res2'), create_resource(self.ctx, self.stack, name='res3')]

    def test_sync_point_create_get(self):
        for res in self.resources:
            sync_point_rsrc = create_sync_point(self.ctx, entity_id=str(res.id), stack_id=self.stack.id, traversal_id=self.stack.current_traversal)
            ret_sync_point_rsrc = db_api.sync_point_get(self.ctx, sync_point_rsrc.entity_id, sync_point_rsrc.traversal_id, sync_point_rsrc.is_update)
            self.assertIsNotNone(ret_sync_point_rsrc)
            self.assertEqual(sync_point_rsrc.entity_id, ret_sync_point_rsrc.entity_id)
            self.assertEqual(sync_point_rsrc.traversal_id, ret_sync_point_rsrc.traversal_id)
            self.assertEqual(sync_point_rsrc.is_update, ret_sync_point_rsrc.is_update)
            self.assertEqual(sync_point_rsrc.atomic_key, ret_sync_point_rsrc.atomic_key)
            self.assertEqual(sync_point_rsrc.stack_id, ret_sync_point_rsrc.stack_id)
            self.assertEqual(sync_point_rsrc.input_data, ret_sync_point_rsrc.input_data)
        sync_point_stack = create_sync_point(self.ctx, entity_id=self.stack.id, stack_id=self.stack.id, traversal_id=self.stack.current_traversal)
        ret_sync_point_stack = db_api.sync_point_get(self.ctx, sync_point_stack.entity_id, sync_point_stack.traversal_id, sync_point_stack.is_update)
        self.assertIsNotNone(ret_sync_point_stack)
        self.assertEqual(sync_point_stack.entity_id, ret_sync_point_stack.entity_id)
        self.assertEqual(sync_point_stack.traversal_id, ret_sync_point_stack.traversal_id)
        self.assertEqual(sync_point_stack.is_update, ret_sync_point_stack.is_update)
        self.assertEqual(sync_point_stack.atomic_key, ret_sync_point_stack.atomic_key)
        self.assertEqual(sync_point_stack.stack_id, ret_sync_point_stack.stack_id)
        self.assertEqual(sync_point_stack.input_data, ret_sync_point_stack.input_data)

    def test_sync_point_update(self):
        sync_point = create_sync_point(self.ctx, entity_id=str(self.resources[0].id), stack_id=self.stack.id, traversal_id=self.stack.current_traversal)
        self.assertEqual({}, sync_point.input_data)
        self.assertEqual(0, sync_point.atomic_key)
        rows_updated = db_api.sync_point_update_input_data(self.ctx, sync_point.entity_id, sync_point.traversal_id, sync_point.is_update, sync_point.atomic_key, {'input_data': '{key: value}'})
        self.assertEqual(1, rows_updated)
        ret_sync_point = db_api.sync_point_get(self.ctx, sync_point.entity_id, sync_point.traversal_id, sync_point.is_update)
        self.assertIsNotNone(ret_sync_point)
        self.assertEqual(1, ret_sync_point.atomic_key)
        self.assertEqual({'input_data': '{key: value}'}, ret_sync_point.input_data)
        rows_updated = db_api.sync_point_update_input_data(self.ctx, ret_sync_point.entity_id, ret_sync_point.traversal_id, ret_sync_point.is_update, ret_sync_point.atomic_key, {'input_data': '{key1: value1}'})
        self.assertEqual(1, rows_updated)
        ret_sync_point = db_api.sync_point_get(self.ctx, sync_point.entity_id, sync_point.traversal_id, sync_point.is_update)
        self.assertIsNotNone(ret_sync_point)
        self.assertEqual(2, ret_sync_point.atomic_key)
        self.assertEqual({'input_data': '{key1: value1}'}, ret_sync_point.input_data)

    def test_sync_point_concurrent_update(self):
        sync_point = create_sync_point(self.ctx, entity_id=str(self.resources[0].id), stack_id=self.stack.id, traversal_id=self.stack.current_traversal)
        self.assertEqual({}, sync_point.input_data)
        self.assertEqual(0, sync_point.atomic_key)
        rows_updated = db_api.sync_point_update_input_data(self.ctx, sync_point.entity_id, sync_point.traversal_id, sync_point.is_update, 0, {'input_data': '{key: value}'})
        self.assertEqual(1, rows_updated)
        rows_updated = db_api.sync_point_update_input_data(self.ctx, sync_point.entity_id, sync_point.traversal_id, sync_point.is_update, 0, {'input_data': '{key: value}'})
        self.assertEqual(0, rows_updated)

    def test_sync_point_delete(self):
        for res in self.resources:
            sync_point_rsrc = create_sync_point(self.ctx, entity_id=str(res.id), stack_id=self.stack.id, traversal_id=self.stack.current_traversal)
            self.assertIsNotNone(sync_point_rsrc)
        sync_point_stack = create_sync_point(self.ctx, entity_id=self.stack.id, stack_id=self.stack.id, traversal_id=self.stack.current_traversal)
        self.assertIsNotNone(sync_point_stack)
        rows_deleted = db_api.sync_point_delete_all_by_stack_and_traversal(self.ctx, self.stack.id, self.stack.current_traversal)
        self.assertGreater(rows_deleted, 0)
        self.assertEqual(4, rows_deleted)
        for res in self.resources:
            ret_sync_point_rsrc = db_api.sync_point_get(self.ctx, str(res.id), self.stack.current_traversal, True)
            self.assertIsNone(ret_sync_point_rsrc)
        ret_sync_point_stack = db_api.sync_point_get(self.ctx, self.stack.id, self.stack.current_traversal, True)
        self.assertIsNone(ret_sync_point_stack)

    @mock.patch.object(time, 'sleep')
    def test_syncpoint_create_deadlock(self, sleep):
        with mock.patch('sqlalchemy.orm.Session.add', side_effect=db_exception.DBDeadlock) as add:
            for res in self.resources:
                self.assertRaises(db_exception.DBDeadlock, create_sync_point, self.ctx, entity_id=str(res.id), stack_id=self.stack.id, traversal_id=self.stack.current_traversal)
            self.assertEqual(len(self.resources) * 21, add.call_count)