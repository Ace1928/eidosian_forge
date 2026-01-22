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
class MetadataRefreshServerTest(common.HeatTestCase):

    @mock.patch.object(nova.NovaClientPlugin, 'find_flavor_by_name_or_id', return_value=1)
    @mock.patch.object(glance.GlanceClientPlugin, 'find_image_by_name_or_id', return_value=1)
    @mock.patch.object(server.Server, 'handle_create')
    @mock.patch.object(server.Server, 'check_create_complete')
    @mock.patch.object(server.Server, 'get_attribute', new_callable=mock.Mock)
    def test_FnGetAtt_metadata_update(self, mock_get, mock_check, mock_handle, *args):
        temp = template_format.parse(TEST_TEMPLATE_SERVER)
        template = tmpl.Template(temp, env=environment.Environment({}))
        ctx = utils.dummy_context()
        stack = stk.Stack(ctx, 'test-stack', template, disable_rollback=True)
        stack.store()
        self.stub_KeypairConstraint_validate()
        mock_get.side_effect = ['192.0.2.1', '192.0.2.2']
        stack.create()
        self.assertEqual((stack.CREATE, stack.COMPLETE), stack.state)
        s1 = stack['instance1']
        s2 = stack['instance2']
        md = s1.metadata_get()
        self.assertEqual({u'template_data': '192.0.2.1'}, md)
        new_md = {u'template_data': '192.0.2.2', 'set_by_rsrc': 'orange'}
        s1.metadata_set(new_md)
        md = s1.metadata_get(refresh=True)
        self.assertEqual(new_md, md)
        s2.attributes.reset_resolved_values()
        stk_defn.update_resource_data(stack.defn, s2.name, s2.node_data())
        s1.metadata_update()
        md = s1.metadata_get(refresh=True)
        self.assertEqual(new_md, md)
        mock_get.assert_has_calls([mock.call('networks'), mock.call('networks')])
        self.assertEqual(2, mock_handle.call_count)
        self.assertEqual(2, mock_check.call_count)