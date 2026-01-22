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
class MetadataRefreshTest(common.HeatTestCase):

    @mock.patch.object(nova.NovaClientPlugin, 'find_flavor_by_name_or_id')
    @mock.patch.object(glance.GlanceClientPlugin, 'find_image_by_name_or_id')
    @mock.patch.object(instance.Instance, 'handle_create')
    @mock.patch.object(instance.Instance, 'check_create_complete')
    @mock.patch.object(instance.Instance, '_resolve_attribute')
    def test_FnGetAtt_metadata_updated(self, mock_get, mock_check, mock_handle, *args):
        """Tests that metadata gets updated when FnGetAtt return changes."""
        temp = template_format.parse(TEST_TEMPLATE_METADATA)
        template = tmpl.Template(temp, env=environment.Environment({}))
        ctx = utils.dummy_context()
        stack = stk.Stack(ctx, 'test_stack', template, disable_rollback=True)
        stack.store()
        self.stub_KeypairConstraint_validate()
        mock_get.side_effect = ['10.0.0.1', '10.0.0.2']
        stack.create()
        self.assertEqual((stack.CREATE, stack.COMPLETE), stack.state)
        s2 = stack['S2']
        self.assertEqual((s2.CREATE, s2.COMPLETE), s2.state)
        s1 = stack['S1']
        content = self._get_metadata_content(s1.metadata_get())
        self.assertEqual('s2-ip=10.0.0.1', content)
        s2.attributes.reset_resolved_values()
        s2.metadata_update()
        stk_defn.update_resource_data(stack.defn, s2.name, s2.node_data())
        s1.metadata_update()
        stk_defn.update_resource_data(stack.defn, s1.name, s1.node_data())
        content = self._get_metadata_content(s1.metadata_get())
        self.assertEqual('s2-ip=10.0.0.2', content)
        mock_get.assert_has_calls([mock.call('PublicIp'), mock.call('PublicIp')])
        self.assertEqual(2, mock_handle.call_count)
        self.assertEqual(2, mock_check.call_count)

    @staticmethod
    def _get_metadata_content(m):
        tmp = m['AWS::CloudFormation::Init']['config']['files']
        return tmp['/tmp/random_file']['content']