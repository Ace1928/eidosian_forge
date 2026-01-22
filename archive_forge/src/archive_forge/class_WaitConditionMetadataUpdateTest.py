from unittest import mock
from oslo_serialization import jsonutils
from heat.common import identifier
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import heat_plugin
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine.resources.aws.cfn.wait_condition_handle import (
from heat.engine.resources.aws.ec2 import instance
from heat.engine.resources.openstack.nova import server
from heat.engine import scheduler
from heat.engine import service
from heat.engine import stack as stk
from heat.engine import stk_defn
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests import utils
class WaitConditionMetadataUpdateTest(common.HeatTestCase):

    def setUp(self):
        super(WaitConditionMetadataUpdateTest, self).setUp()
        self.man = service.EngineService('a-host', 'a-topic')

    @mock.patch.object(nova.NovaClientPlugin, 'find_flavor_by_name_or_id')
    @mock.patch.object(glance.GlanceClientPlugin, 'find_image_by_name_or_id')
    @mock.patch.object(heat_plugin.HeatClientPlugin, 'get_heat_cfn_url')
    @mock.patch.object(instance.Instance, 'handle_create')
    @mock.patch.object(instance.Instance, 'check_create_complete')
    @mock.patch.object(scheduler.TaskRunner, '_sleep')
    @mock.patch.object(WaitConditionHandle, 'identifier')
    def test_wait_metadata(self, mock_identifier, mock_sleep, mock_check, mock_handle, mock_get, *args):
        """Tests a wait condition metadata update after a signal call."""
        mock_get.return_value = 'http://server.test:8000/v1'
        temp = template_format.parse(TEST_TEMPLATE_WAIT_CONDITION)
        template = tmpl.Template(temp)
        ctx = utils.dummy_context()
        stack = stk.Stack(ctx, 'test-stack', template, disable_rollback=True)
        stack.store()
        self.stub_KeypairConstraint_validate()
        res_id = identifier.ResourceIdentifier('test_tenant_id', stack.name, stack.id, '', 'WH')
        mock_identifier.return_value = res_id
        watch = stack['WC']
        inst = stack['S2']
        self.run_empty = True

        def check_empty(sleep_time):
            self.assertEqual('{}', watch.FnGetAtt('Data'))
            self.assertIsNone(inst.metadata_get()['test'])

        def update_metadata(unique_id, data, reason):
            self.man.resource_signal(ctx, dict(stack.identifier()), 'WH', {'Data': data, 'Reason': reason, 'Status': 'SUCCESS', 'UniqueId': unique_id}, sync_call=True)

        def post_success(sleep_time):
            update_metadata('123', 'foo', 'bar')

        def side_effect_popper(sleep_time):
            wh = stack['WH']
            if wh.status == wh.IN_PROGRESS:
                return
            elif self.run_empty:
                self.run_empty = False
                check_empty(sleep_time)
            else:
                post_success(sleep_time)
        mock_sleep.side_effect = side_effect_popper
        stack.create()
        self.assertEqual((stack.CREATE, stack.COMPLETE), stack.state)
        self.assertEqual('{"123": "foo"}', watch.FnGetAtt('Data'))
        self.assertEqual('{"123": "foo"}', inst.metadata_get()['test'])
        update_metadata('456', 'blarg', 'wibble')
        self.assertEqual({'123': 'foo', '456': 'blarg'}, jsonutils.loads(watch.FnGetAtt('Data')))
        self.assertEqual('{"123": "foo"}', inst.metadata_get()['test'])
        self.assertEqual({'123': 'foo', '456': 'blarg'}, jsonutils.loads(inst.metadata_get(refresh=True)['test']))
        self.assertEqual(2, mock_handle.call_count)
        self.assertEqual(2, mock_check.call_count)