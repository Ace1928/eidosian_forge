from unittest import mock
import uuid
import eventlet.queue
from oslo_config import cfg
from oslo_messaging import conffixture
from oslo_messaging.rpc import dispatcher
from heat.common import context
from heat.common import environment_util as env_util
from heat.common import exception
from heat.common import messaging
from heat.common import service_utils
from heat.common import template_format
from heat.db import api as db_api
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine.clients.os import swift
from heat.engine import environment
from heat.engine import resource
from heat.engine import service
from heat.engine import stack
from heat.engine import stack_lock
from heat.engine import template as templatem
from heat.objects import stack as stack_object
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
class ServiceStackUpdatePreviewTest(common.HeatTestCase):
    old_tmpl = '\nheat_template_version: 2014-10-16\nresources:\n  web_server:\n    type: OS::Nova::Server\n    properties:\n      image: F17-x86_64-gold\n      flavor: m1.large\n      key_name: test\n      user_data: wordpress\n    '
    new_tmpl = '\nheat_template_version: 2014-10-16\nresources:\n  web_server:\n    type: OS::Nova::Server\n    properties:\n      image: F17-x86_64-gold\n      flavor: m1.large\n      key_name: test\n      user_data: wordpress\n  password:\n    type: OS::Heat::RandomString\n    properties:\n      length: 8\n    '

    def setUp(self):
        super(ServiceStackUpdatePreviewTest, self).setUp()
        self.ctx = utils.dummy_context()
        self.man = service.EngineService('a-host', 'a-topic')
        self.man.thread_group_mgr = tools.DummyThreadGroupManager()

    def _test_stack_update_preview(self, orig_template, new_template, environment_files=None):
        stack_name = 'service_update_test_stack_preview'
        params = {'foo': 'bar'}

        def side_effect(*args):
            return 2 if args[0] == 'm1.small' else 1
        self.patchobject(nova.NovaClientPlugin, 'find_flavor_by_name_or_id', side_effect=side_effect)
        self.patchobject(glance.GlanceClientPlugin, 'find_image_by_name_or_id', return_value=1)
        old_stack = tools.get_stack(stack_name, self.ctx, template=orig_template)
        sid = old_stack.store()
        old_stack.set_stack_user_project_id('1234')
        s = stack_object.Stack.get_by_id(self.ctx, sid)
        stk = tools.get_stack(stack_name, self.ctx, template=new_template)
        mock_stack = self.patchobject(stack, 'Stack', return_value=stk)
        mock_load = self.patchobject(stack.Stack, 'load', return_value=old_stack)
        mock_tmpl = self.patchobject(templatem, 'Template', return_value=stk.t)
        mock_env = self.patchobject(environment, 'Environment', return_value=stk.env)
        mock_validate = self.patchobject(stk, 'validate', return_value=None)
        mock_merge = self.patchobject(env_util, 'merge_environments')
        self.patchobject(resource.Resource, '_resolve_any_attribute', return_value=None)
        api_args = {'timeout_mins': 60, rpc_api.PARAM_CONVERGE: False}
        result = self.man.preview_update_stack(self.ctx, old_stack.identifier(), new_template, params, None, api_args, environment_files=environment_files)
        mock_stack.assert_called_once_with(self.ctx, stk.name, stk.t, convergence=False, current_traversal=old_stack.current_traversal, prev_raw_template_id=None, current_deps=None, disable_rollback=True, nested_depth=0, owner_id=None, parent_resource=None, stack_user_project_id='1234', strict_validate=True, tenant_id='test_tenant_id', timeout_mins=60, user_creds_id=u'1', username='test_username', converge=False)
        mock_load.assert_called_once_with(self.ctx, stack=s)
        mock_tmpl.assert_called_once_with(new_template, files=None)
        mock_env.assert_called_once_with(params)
        mock_validate.assert_called_once_with()
        if environment_files:
            mock_merge.assert_called_once_with(environment_files, None, params, mock.ANY)
        return result

    def test_stack_update_preview_added_unchanged(self):
        result = self._test_stack_update_preview(self.old_tmpl, self.new_tmpl)
        added = [x for x in result['added']][0]
        self.assertEqual('password', added['resource_name'])
        unchanged = [x for x in result['unchanged']][0]
        self.assertEqual('web_server', unchanged['resource_name'])
        self.assertNotEqual('None', unchanged['resource_identity']['stack_id'])
        empty_sections = ('deleted', 'replaced', 'updated')
        for section in empty_sections:
            section_contents = [x for x in result[section]]
            self.assertEqual([], section_contents)

    def test_stack_update_preview_replaced(self):
        new_tmpl = self.old_tmpl.replace('test', 'test2')
        result = self._test_stack_update_preview(self.old_tmpl, new_tmpl)
        replaced = [x for x in result['replaced']][0]
        self.assertEqual('web_server', replaced['resource_name'])
        empty_sections = ('added', 'deleted', 'unchanged', 'updated')
        for section in empty_sections:
            section_contents = [x for x in result[section]]
            self.assertEqual([], section_contents)

    def test_stack_update_preview_replaced_type(self):
        new_tmpl = self.old_tmpl.replace('OS::Nova::Server', 'OS::Heat::None')
        result = self._test_stack_update_preview(self.old_tmpl, new_tmpl)
        replaced = [x for x in result['replaced']][0]
        self.assertEqual('web_server', replaced['resource_name'])
        empty_sections = ('added', 'deleted', 'unchanged', 'updated')
        for section in empty_sections:
            section_contents = [x for x in result[section]]
            self.assertEqual([], section_contents)

    def test_stack_update_preview_updated(self):
        new_tmpl = self.old_tmpl.replace('m1.large', 'm1.small')
        result = self._test_stack_update_preview(self.old_tmpl, new_tmpl)
        updated = [x for x in result['updated']][0]
        self.assertEqual('web_server', updated['resource_name'])
        empty_sections = ('added', 'deleted', 'unchanged', 'replaced')
        for section in empty_sections:
            section_contents = [x for x in result[section]]
            self.assertEqual([], section_contents)

    def test_stack_update_preview_deleted(self):
        result = self._test_stack_update_preview(self.new_tmpl, self.old_tmpl)
        deleted = [x for x in result['deleted']][0]
        self.assertEqual('password', deleted['resource_name'])
        unchanged = [x for x in result['unchanged']][0]
        self.assertEqual('web_server', unchanged['resource_name'])
        empty_sections = ('added', 'updated', 'replaced')
        for section in empty_sections:
            section_contents = [x for x in result[section]]
            self.assertEqual([], section_contents)

    def test_stack_update_preview_with_environment_files(self):
        environment_files = ['env_1']
        self._test_stack_update_preview(self.old_tmpl, self.new_tmpl, environment_files=environment_files)

    def test_reset_stack_and_resources_in_progress(self):

        def mock_stack_resource(name, action, status):
            rs = mock.MagicMock()
            rs.name = name
            rs.action = action
            rs.status = status
            rs.IN_PROGRESS = 'IN_PROGRESS'
            rs.FAILED = 'FAILED'

            def mock_resource_state_set(a, s, reason='engine_down'):
                rs.status = s
                rs.action = a
                rs.status_reason = reason
            rs.state_set = mock_resource_state_set
            return rs
        stk_name = 'test_stack'
        stk = tools.get_stack(stk_name, self.ctx)
        stk.action = 'CREATE'
        stk.status = 'IN_PROGRESS'
        resources = {'r1': mock_stack_resource('r1', 'UPDATE', 'COMPLETE'), 'r2': mock_stack_resource('r2', 'UPDATE', 'IN_PROGRESS'), 'r3': mock_stack_resource('r3', 'UPDATE', 'FAILED')}
        stk._resources = resources
        reason = 'Test resetting stack and resources in progress'
        stk.reset_stack_and_resources_in_progress(reason)
        self.assertEqual('FAILED', stk.status)
        self.assertEqual('COMPLETE', stk.resources.get('r1').status)
        self.assertEqual('FAILED', stk.resources.get('r2').status)
        self.assertEqual('FAILED', stk.resources.get('r3').status)