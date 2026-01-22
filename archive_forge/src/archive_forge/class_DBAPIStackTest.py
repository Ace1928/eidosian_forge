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
class DBAPIStackTest(common.HeatTestCase):

    def setUp(self):
        super(DBAPIStackTest, self).setUp()
        self.ctx = utils.dummy_context()
        self.template = create_raw_template(self.ctx)
        self.user_creds = create_user_creds(self.ctx)

    def test_stack_create(self):
        stack = create_stack(self.ctx, self.template, self.user_creds)
        self.assertIsNotNone(stack.id)
        self.assertEqual(12, len(stack.name))
        self.assertEqual(self.template.id, stack.raw_template_id)
        self.assertEqual(self.ctx.username, stack.username)
        self.assertEqual(self.ctx.tenant_id, stack.tenant)
        self.assertEqual('create', stack.action)
        self.assertEqual('complete', stack.status)
        self.assertEqual('create_complete', stack.status_reason)
        self.assertEqual({}, stack.parameters)
        self.assertEqual(self.user_creds['id'], stack.user_creds_id)
        self.assertIsNone(stack.owner_id)
        self.assertEqual('60', stack.timeout)
        self.assertFalse(stack.disable_rollback)

    def test_stack_delete(self):
        stack = create_stack(self.ctx, self.template, self.user_creds)
        stack_id = stack.id
        resource = create_resource(self.ctx, stack)
        db_api.stack_delete(self.ctx, stack_id)
        self.assertIsNone(db_api.stack_get(self.ctx, stack_id, show_deleted=False))
        self.assertRaises(exception.NotFound, db_api.resource_get, self.ctx, resource.id)
        self.assertRaises(exception.NotFound, db_api.stack_delete, self.ctx, stack_id)
        ret_stack = db_api.stack_get(self.ctx, stack_id, show_deleted=True)
        self.assertIsNotNone(ret_stack)
        self.assertEqual(stack_id, ret_stack.id)
        self.assertEqual(12, len(ret_stack.name))
        self.assertRaises(exception.NotFound, db_api.resource_get, self.ctx, resource.id)

    def test_stack_update(self):
        stack = create_stack(self.ctx, self.template, self.user_creds)
        values = {'name': 'db_test_stack_name2', 'action': 'update', 'status': 'failed', 'status_reason': 'update_failed', 'timeout': '90', 'current_traversal': 'another-dummy-uuid'}
        db_api.stack_update(self.ctx, stack.id, values)
        stack = db_api.stack_get(self.ctx, stack.id)
        self.assertEqual('db_test_stack_name2', stack.name)
        self.assertEqual('update', stack.action)
        self.assertEqual('failed', stack.status)
        self.assertEqual('update_failed', stack.status_reason)
        self.assertEqual(90, stack.timeout)
        self.assertEqual('another-dummy-uuid', stack.current_traversal)
        self.assertRaises(exception.NotFound, db_api.stack_update, self.ctx, UUID2, values)

    def test_stack_update_matches_traversal_id(self):
        stack = create_stack(self.ctx, self.template, self.user_creds)
        values = {'current_traversal': 'another-dummy-uuid'}
        updated = db_api.stack_update(self.ctx, stack.id, values, exp_trvsl='dummy-uuid')
        self.assertTrue(updated)
        stack = db_api.stack_get(self.ctx, stack.id)
        self.assertEqual('another-dummy-uuid', stack.current_traversal)
        matching_uuid = 'another-dummy-uuid'
        updated = db_api.stack_update(self.ctx, stack.id, values, exp_trvsl=matching_uuid)
        self.assertTrue(updated)
        diff_uuid = 'some-other-dummy-uuid'
        updated = db_api.stack_update(self.ctx, stack.id, values, exp_trvsl=diff_uuid)
        self.assertFalse(updated)

    @mock.patch.object(time, 'sleep')
    def test_stack_update_retries_on_deadlock(self, sleep):
        stack = create_stack(self.ctx, self.template, self.user_creds)
        with mock.patch('sqlalchemy.orm.query.Query.update', side_effect=db_exception.DBDeadlock) as mock_update:
            self.assertRaises(db_exception.DBDeadlock, db_api.stack_update, self.ctx, stack.id, {})
            self.assertEqual(21, mock_update.call_count)

    def test_stack_set_status_release_lock(self):
        stack = create_stack(self.ctx, self.template, self.user_creds)
        values = {'name': 'db_test_stack_name2', 'action': 'update', 'status': 'failed', 'status_reason': 'update_failed', 'timeout': '90', 'current_traversal': 'another-dummy-uuid'}
        db_api.stack_lock_create(self.ctx, stack.id, UUID1)
        observed = db_api.persist_state_and_release_lock(self.ctx, stack.id, UUID1, values)
        self.assertIsNone(observed)
        stack = db_api.stack_get(self.ctx, stack.id)
        self.assertEqual('db_test_stack_name2', stack.name)
        self.assertEqual('update', stack.action)
        self.assertEqual('failed', stack.status)
        self.assertEqual('update_failed', stack.status_reason)
        self.assertEqual(90, stack.timeout)
        self.assertEqual('another-dummy-uuid', stack.current_traversal)

    def test_stack_set_status_release_lock_failed(self):
        stack = create_stack(self.ctx, self.template, self.user_creds)
        values = {'name': 'db_test_stack_name2', 'action': 'update', 'status': 'failed', 'status_reason': 'update_failed', 'timeout': '90', 'current_traversal': 'another-dummy-uuid'}
        db_api.stack_lock_create(self.ctx, stack.id, UUID2)
        observed = db_api.persist_state_and_release_lock(self.ctx, stack.id, UUID1, values)
        self.assertTrue(observed)

    def test_stack_set_status_failed_release_lock(self):
        stack = create_stack(self.ctx, self.template, self.user_creds)
        values = {'name': 'db_test_stack_name2', 'action': 'update', 'status': 'failed', 'status_reason': 'update_failed', 'timeout': '90', 'current_traversal': 'another-dummy-uuid'}
        db_api.stack_lock_create(self.ctx, stack.id, UUID1)
        observed = db_api.persist_state_and_release_lock(self.ctx, UUID2, UUID1, values)
        self.assertTrue(observed)

    def test_stack_get_returns_a_stack(self):
        stack = create_stack(self.ctx, self.template, self.user_creds)
        ret_stack = db_api.stack_get(self.ctx, stack.id, show_deleted=False)
        self.assertIsNotNone(ret_stack)
        self.assertEqual(stack.id, ret_stack.id)
        self.assertEqual(12, len(ret_stack.name))

    def test_stack_get_returns_none_if_stack_does_not_exist(self):
        stack = db_api.stack_get(self.ctx, UUID1, show_deleted=False)
        self.assertIsNone(stack)

    def test_stack_get_returns_none_if_tenant_id_does_not_match(self):
        stack = create_stack(self.ctx, self.template, self.user_creds)
        self.ctx.project_id = 'abc'
        stack = db_api.stack_get(self.ctx, UUID1, show_deleted=False)
        self.assertIsNone(stack)

    def test_stack_get_tenant_is_stack_user_project_id(self):
        stack = create_stack(self.ctx, self.template, self.user_creds, stack_user_project_id='astackuserproject')
        self.ctx.project_id = 'astackuserproject'
        ret_stack = db_api.stack_get(self.ctx, stack.id, show_deleted=False)
        self.assertIsNotNone(ret_stack)
        self.assertEqual(stack.id, ret_stack.id)
        self.assertEqual(12, len(ret_stack.name))

    def test_stack_get_can_return_a_stack_from_different_tenant(self):
        stack = create_stack(self.ctx, self.template, self.user_creds)
        admin_ctx = utils.dummy_context(user='admin_username', tenant_id='admin_tenant', is_admin=True)
        ret_stack = db_api.stack_get(admin_ctx, stack.id, show_deleted=False)
        self.assertEqual(stack.id, ret_stack.id)
        self.assertEqual(12, len(ret_stack.name))

    def test_stack_get_by_name(self):
        stack = create_stack(self.ctx, self.template, self.user_creds)
        ret_stack = db_api.stack_get_by_name(self.ctx, stack.name)
        self.assertIsNotNone(ret_stack)
        self.assertEqual(stack.id, ret_stack.id)
        self.assertEqual(12, len(ret_stack.name))
        self.assertIsNone(db_api.stack_get_by_name(self.ctx, 'abc'))
        self.ctx.project_id = 'abc'
        self.assertIsNone(db_api.stack_get_by_name(self.ctx, 'abc'))

    def test_stack_get_all(self):
        values = [{'name': 'stack1'}, {'name': 'stack2'}, {'name': 'stack3'}, {'name': 'stack4'}]
        [create_stack(self.ctx, self.template, self.user_creds, **val) for val in values]
        ret_stacks = db_api.stack_get_all(self.ctx)
        self.assertEqual(4, len(ret_stacks))
        names = [ret_stack.name for ret_stack in ret_stacks]
        [self.assertIn(val['name'], names) for val in values]

    def test_stack_get_all_by_owner_id(self):
        parent_stack1 = create_stack(self.ctx, self.template, self.user_creds)
        parent_stack2 = create_stack(self.ctx, self.template, self.user_creds)
        values = [{'owner_id': parent_stack1.id}, {'owner_id': parent_stack1.id}, {'owner_id': parent_stack2.id}, {'owner_id': parent_stack2.id}]
        [create_stack(self.ctx, self.template, self.user_creds, **val) for val in values]
        stack1_children = db_api.stack_get_all_by_owner_id(self.ctx, parent_stack1.id)
        self.assertEqual(2, len(stack1_children))
        stack2_children = db_api.stack_get_all_by_owner_id(self.ctx, parent_stack2.id)
        self.assertEqual(2, len(stack2_children))

    def test_stack_get_all_by_root_owner_id(self):
        parent_stack1 = create_stack(self.ctx, self.template, self.user_creds)
        parent_stack2 = create_stack(self.ctx, self.template, self.user_creds)
        for i in range(3):
            lvl1_st = create_stack(self.ctx, self.template, self.user_creds, owner_id=parent_stack1.id)
            for j in range(2):
                create_stack(self.ctx, self.template, self.user_creds, owner_id=lvl1_st.id)
        for i in range(2):
            lvl1_st = create_stack(self.ctx, self.template, self.user_creds, owner_id=parent_stack2.id)
            for j in range(4):
                lvl2_st = create_stack(self.ctx, self.template, self.user_creds, owner_id=lvl1_st.id)
                for k in range(3):
                    create_stack(self.ctx, self.template, self.user_creds, owner_id=lvl2_st.id)
        stack1_children = db_api.stack_get_all_by_root_owner_id(self.ctx, parent_stack1.id)
        self.assertEqual(9, len(stack1_children))
        stack2_children = db_api.stack_get_all_by_root_owner_id(self.ctx, parent_stack2.id)
        self.assertEqual(34, len(list(stack2_children)))

    def test_stack_get_all_with_regular_tenant(self):
        values = [{'tenant': UUID1}, {'tenant': UUID1}, {'tenant': UUID2}, {'tenant': UUID2}, {'tenant': UUID2}]
        [create_stack(self.ctx, self.template, self.user_creds, **val) for val in values]
        self.ctx.project_id = UUID1
        stacks = db_api.stack_get_all(self.ctx)
        self.assertEqual(2, len(stacks))
        self.ctx.project_id = UUID2
        stacks = db_api.stack_get_all(self.ctx)
        self.assertEqual(3, len(stacks))
        self.ctx.project_id = UUID3
        self.assertEqual([], db_api.stack_get_all(self.ctx))

    def test_stack_get_all_with_admin_context(self):
        values = [{'tenant': UUID1}, {'tenant': UUID1}, {'tenant': UUID2}, {'tenant': UUID2}, {'tenant': UUID2}]
        [create_stack(self.ctx, self.template, self.user_creds, **val) for val in values]
        admin_ctx = utils.dummy_context(user='admin_user', tenant_id='admin_tenant', is_admin=True)
        stacks = db_api.stack_get_all(admin_ctx)
        self.assertEqual(5, len(stacks))

    def test_stack_count_all_with_regular_tenant(self):
        values = [{'tenant': UUID1}, {'tenant': UUID1}, {'tenant': UUID2}, {'tenant': UUID2}, {'tenant': UUID2}]
        [create_stack(self.ctx, self.template, self.user_creds, **val) for val in values]
        self.ctx.project_id = UUID1
        self.assertEqual(2, db_api.stack_count_all(self.ctx))
        self.ctx.project_id = UUID2
        self.assertEqual(3, db_api.stack_count_all(self.ctx))

    def test_stack_count_all_with_admin_context(self):
        values = [{'tenant': UUID1}, {'tenant': UUID1}, {'tenant': UUID2}, {'tenant': UUID2}, {'tenant': UUID2}]
        [create_stack(self.ctx, self.template, self.user_creds, **val) for val in values]
        admin_ctx = utils.dummy_context(user='admin_user', tenant_id='admin_tenant', is_admin=True)
        self.assertEqual(5, db_api.stack_count_all(admin_ctx))

    def test_purge_deleted(self):
        now = timeutils.utcnow()
        delta = datetime.timedelta(seconds=3600 * 7)
        deleted = [now - delta * i for i in range(1, 6)]
        tmpl_files = [template_files.TemplateFiles({'foo': 'file contents %d' % i}) for i in range(5)]
        [tmpl_file.store(self.ctx) for tmpl_file in tmpl_files]
        templates = [create_raw_template(self.ctx, files_id=tmpl_files[i].files_id) for i in range(5)]
        creds = [create_user_creds(self.ctx) for i in range(5)]
        stacks = [create_stack(self.ctx, templates[i], creds[i], deleted_at=deleted[i]) for i in range(5)]
        resources = [create_resource(self.ctx, stacks[i]) for i in range(5)]
        events = [create_event(self.ctx, stack_id=stacks[i].id) for i in range(5)]
        db_api.purge_deleted(age=1, granularity='days')
        admin_ctx = utils.dummy_context(is_admin=True)
        self._deleted_stack_existance(admin_ctx, stacks, resources, events, tmpl_files, (0, 1, 2), (3, 4))
        db_api.purge_deleted(age=22, granularity='hours')
        self._deleted_stack_existance(admin_ctx, stacks, resources, events, tmpl_files, (0, 1, 2), (3, 4))
        db_api.purge_deleted(age=1100, granularity='minutes')
        self._deleted_stack_existance(admin_ctx, stacks, resources, events, tmpl_files, (0, 1), (2, 3, 4))
        db_api.purge_deleted(age=3600, granularity='seconds')
        self._deleted_stack_existance(admin_ctx, stacks, resources, events, tmpl_files, (), (0, 1, 2, 3, 4))
        self.assertRaises(exception.Error, db_api.purge_deleted, -1, 'seconds')

    def test_purge_project_deleted(self):
        now = timeutils.utcnow()
        delta = datetime.timedelta(seconds=3600 * 7)
        deleted = [now - delta * i for i in range(1, 6)]
        tmpl_files = [template_files.TemplateFiles({'foo': 'file contents %d' % i}) for i in range(5)]
        [tmpl_file.store(self.ctx) for tmpl_file in tmpl_files]
        templates = [create_raw_template(self.ctx, files_id=tmpl_files[i].files_id) for i in range(5)]
        values = [{'tenant': UUID1}, {'tenant': UUID1}, {'tenant': UUID1}, {'tenant': UUID2}, {'tenant': UUID2}]
        creds = [create_user_creds(self.ctx) for i in range(5)]
        stacks = [create_stack(self.ctx, templates[i], creds[i], deleted_at=deleted[i], **values[i]) for i in range(5)]
        resources = [create_resource(self.ctx, stacks[i]) for i in range(5)]
        events = [create_event(self.ctx, stack_id=stacks[i].id) for i in range(5)]
        db_api.purge_deleted(age=1, granularity='days', project_id=UUID1)
        admin_ctx = utils.dummy_context(is_admin=True)
        self._deleted_stack_existance(admin_ctx, stacks, resources, events, tmpl_files, (0, 1, 2, 3, 4), ())
        db_api.purge_deleted(age=22, granularity='hours', project_id=UUID1)
        self._deleted_stack_existance(admin_ctx, stacks, resources, events, tmpl_files, (0, 1, 2, 3, 4), ())
        db_api.purge_deleted(age=1100, granularity='minutes', project_id=UUID1)
        self._deleted_stack_existance(admin_ctx, stacks, resources, events, tmpl_files, (0, 1, 3, 4), (2,))
        db_api.purge_deleted(age=30, granularity='hours', project_id=UUID2)
        self._deleted_stack_existance(admin_ctx, stacks, resources, events, tmpl_files, (0, 1, 3), (2, 4))
        db_api.purge_deleted(age=3600, granularity='seconds', project_id=UUID1)
        self._deleted_stack_existance(admin_ctx, stacks, resources, events, tmpl_files, (3,), (0, 1, 2, 4))
        db_api.purge_deleted(age=3600, granularity='seconds', project_id=UUID2)
        self._deleted_stack_existance(admin_ctx, stacks, resources, events, tmpl_files, (), (0, 1, 2, 3, 4))

    def test_purge_deleted_prev_raw_template(self):
        now = timeutils.utcnow()
        templates = [create_raw_template(self.ctx) for i in range(2)]
        stacks = [create_stack(self.ctx, templates[0], create_user_creds(self.ctx), deleted_at=now - datetime.timedelta(seconds=10), prev_raw_template=templates[1])]
        db_api.purge_deleted(age=3600, granularity='seconds')
        ctx = utils.dummy_context(is_admin=True)
        self.assertIsNotNone(db_api.stack_get(ctx, stacks[0].id, show_deleted=True))
        self.assertIsNotNone(db_api.raw_template_get(ctx, templates[1].id))
        stacks = [create_stack(self.ctx, templates[0], create_user_creds(self.ctx), deleted_at=now - datetime.timedelta(seconds=10), prev_raw_template=templates[1], tenant=UUID1)]
        db_api.purge_deleted(age=3600, granularity='seconds', project_id=UUID1)
        self.assertIsNotNone(db_api.stack_get(ctx, stacks[0].id, show_deleted=True))
        self.assertIsNotNone(db_api.raw_template_get(ctx, templates[1].id))
        db_api.purge_deleted(age=0, granularity='seconds', project_id=UUID2)
        self.assertIsNotNone(db_api.stack_get(ctx, stacks[0].id, show_deleted=True))
        self.assertIsNotNone(db_api.raw_template_get(ctx, templates[1].id))

    def test_dont_purge_shared_raw_template_files(self):
        now = timeutils.utcnow()
        delta = datetime.timedelta(seconds=3600 * 7)
        deleted = [now - delta * i for i in range(1, 6)]
        tmpl_files = [template_files.TemplateFiles({'foo': 'more file contents'}) for i in range(3)]
        [tmpl_file.store(self.ctx) for tmpl_file in tmpl_files]
        templates = [create_raw_template(self.ctx, files_id=tmpl_files[i % 3].files_id) for i in range(5)]
        creds = [create_user_creds(self.ctx) for i in range(5)]
        [create_stack(self.ctx, templates[i], creds[i], deleted_at=deleted[i]) for i in range(5)]
        db_api.purge_deleted(age=15, granularity='hours')
        self.assertIsNotNone(db_api.raw_template_files_get(self.ctx, tmpl_files[0].files_id))
        self.assertIsNotNone(db_api.raw_template_files_get(self.ctx, tmpl_files[1].files_id))
        self.assertRaises(exception.NotFound, db_api.raw_template_files_get, self.ctx, tmpl_files[2].files_id)

    def test_dont_purge_project_shared_raw_template_files(self):
        now = timeutils.utcnow()
        delta = datetime.timedelta(seconds=3600 * 7)
        deleted = [now - delta * i for i in range(1, 6)]
        tmpl_files = [template_files.TemplateFiles({'foo': 'more file contents'}) for i in range(3)]
        [tmpl_file.store(self.ctx) for tmpl_file in tmpl_files]
        templates = [create_raw_template(self.ctx, files_id=tmpl_files[i % 3].files_id) for i in range(5)]
        creds = [create_user_creds(self.ctx) for i in range(5)]
        [create_stack(self.ctx, templates[i], creds[i], deleted_at=deleted[i], tenant=UUID1) for i in range(5)]
        db_api.purge_deleted(age=0, granularity='seconds', project_id=UUID3)
        self.assertIsNotNone(db_api.raw_template_files_get(self.ctx, tmpl_files[0].files_id))
        self.assertIsNotNone(db_api.raw_template_files_get(self.ctx, tmpl_files[1].files_id))
        self.assertIsNotNone(db_api.raw_template_files_get(self.ctx, tmpl_files[2].files_id))
        db_api.purge_deleted(age=15, granularity='hours', project_id=UUID1)
        self.assertIsNotNone(db_api.raw_template_files_get(self.ctx, tmpl_files[0].files_id))
        self.assertIsNotNone(db_api.raw_template_files_get(self.ctx, tmpl_files[1].files_id))
        self.assertRaises(exception.NotFound, db_api.raw_template_files_get, self.ctx, tmpl_files[2].files_id)

    def _deleted_stack_existance(self, ctx, stacks, resources, events, tmpl_files, existing, deleted):
        for s in existing:
            self.assertIsNotNone(db_api.stack_get(ctx, stacks[s].id, show_deleted=True))
            self.assertIsNotNone(db_api.raw_template_files_get(ctx, tmpl_files[s].files_id))
            self.assertIsNotNone(db_api.resource_get(ctx, resources[s].id))
            with db_api.context_manager.reader.using(ctx):
                self.assertIsNotNone(ctx.session.get(models.Event, events[s].id))
                self.assertIsNotNone(ctx.session.query(models.ResourcePropertiesData).filter_by(id=resources[s].rsrc_prop_data.id).first())
                self.assertIsNotNone(ctx.session.query(models.ResourcePropertiesData).filter_by(id=events[s].rsrc_prop_data.id).first())
        for s in deleted:
            self.assertIsNone(db_api.stack_get(ctx, stacks[s].id, show_deleted=True))
            rt_id = stacks[s].raw_template_id
            self.assertRaises(exception.NotFound, db_api.raw_template_get, ctx, rt_id)
            self.assertEqual({}, db_api.resource_get_all_by_stack(ctx, stacks[s].id))
            self.assertRaises(exception.NotFound, db_api.raw_template_files_get, ctx, tmpl_files[s].files_id)
            self.assertEqual([], db_api.event_get_all_by_stack(ctx, stacks[s].id))
            with db_api.context_manager.reader.using(ctx):
                self.assertIsNone(ctx.session.get(models.Event, events[s].id))
                self.assertIsNone(ctx.session.query(models.ResourcePropertiesData).filter_by(id=resources[s].rsrc_prop_data.id).first())
                self.assertIsNone(ctx.session.query(models.ResourcePropertiesData).filter_by(id=events[s].rsrc_prop_data.id).first())
            self.assertEqual([], db_api.event_get_all_by_stack(ctx, stacks[s].id))
            self.assertIsNone(db_api.user_creds_get(self.ctx, stacks[s].user_creds_id))

    def test_purge_deleted_batch_arg(self):
        now = timeutils.utcnow()
        delta = datetime.timedelta(seconds=3600)
        deleted = now - delta
        for i in range(7):
            create_stack(self.ctx, self.template, self.user_creds, deleted_at=deleted)
        with mock.patch('heat.db.api._purge_stacks') as mock_ps:
            db_api.purge_deleted(age=0, batch_size=2)
            self.assertEqual(4, mock_ps.call_count)

    def test_stack_get_root_id(self):
        root = create_stack(self.ctx, self.template, self.user_creds, name='root stack')
        child_1 = create_stack(self.ctx, self.template, self.user_creds, name='child 1 stack', owner_id=root.id)
        child_2 = create_stack(self.ctx, self.template, self.user_creds, name='child 2 stack', owner_id=child_1.id)
        child_3 = create_stack(self.ctx, self.template, self.user_creds, name='child 3 stack', owner_id=child_2.id)
        self.assertEqual(root.id, db_api.stack_get_root_id(self.ctx, child_3.id))
        self.assertEqual(root.id, db_api.stack_get_root_id(self.ctx, child_2.id))
        self.assertEqual(root.id, db_api.stack_get_root_id(self.ctx, root.id))
        self.assertEqual(root.id, db_api.stack_get_root_id(self.ctx, child_1.id))
        self.assertIsNone(db_api.stack_get_root_id(self.ctx, 'non existent stack'))

    def test_stack_count_total_resources(self):

        def add_resources(stack, count, root_stack_id):
            for i in range(count):
                create_resource(self.ctx, stack, False, name='%s-%s' % (stack.name, i), root_stack_id=root_stack_id)
        root = create_stack(self.ctx, self.template, self.user_creds, name='root stack')
        s_1 = create_stack(self.ctx, self.template, self.user_creds, name='s_1', owner_id=root.id)
        s_1_1 = create_stack(self.ctx, self.template, self.user_creds, name='s_1_1', owner_id=s_1.id)
        s_1_2 = create_stack(self.ctx, self.template, self.user_creds, name='s_1_2', owner_id=s_1.id)
        s_1_3 = create_stack(self.ctx, self.template, self.user_creds, name='s_1_3', owner_id=s_1.id)
        s_2 = create_stack(self.ctx, self.template, self.user_creds, name='s_2', owner_id=root.id)
        s_2_1 = create_stack(self.ctx, self.template, self.user_creds, name='s_2_1', owner_id=s_2.id)
        s_2_1_1 = create_stack(self.ctx, self.template, self.user_creds, name='s_2_1_1', owner_id=s_2_1.id)
        s_2_1_1_1 = create_stack(self.ctx, self.template, self.user_creds, name='s_2_1_1_1', owner_id=s_2_1_1.id)
        s_3 = create_stack(self.ctx, self.template, self.user_creds, name='s_3', owner_id=root.id)
        s_4 = create_stack(self.ctx, self.template, self.user_creds, name='s_4', owner_id=root.id)
        add_resources(root, 3, root.id)
        add_resources(s_1, 2, root.id)
        add_resources(s_1_1, 4, root.id)
        add_resources(s_1_2, 5, root.id)
        add_resources(s_1_3, 6, root.id)
        add_resources(s_2, 1, root.id)
        add_resources(s_2_1_1_1, 1, root.id)
        add_resources(s_3, 4, root.id)
        self.assertEqual(26, db_api.stack_count_total_resources(self.ctx, root.id))
        self.assertEqual(0, db_api.stack_count_total_resources(self.ctx, s_4.id))
        self.assertEqual(0, db_api.stack_count_total_resources(self.ctx, 'asdf'))
        self.assertEqual(0, db_api.stack_count_total_resources(self.ctx, None))