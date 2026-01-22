import copy
from unittest import mock
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine.resources.openstack.heat import resource_group
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
class ResourceGroupAttrFallbackTest(ResourceGroupAttrTest):

    def _stub_get_attr(self, resg, refids, attrs):
        resg.get_output = mock.Mock(side_effect=exception.NotFound)

        def make_fake_res(idx):
            fr = mock.Mock()
            fr.stack = resg.stack
            fr.FnGetRefId.return_value = refids[idx]
            fr.FnGetAtt.return_value = attrs[idx]
            return fr
        fake_res = {str(i): make_fake_res(i) for i in refids}
        resg.nested = mock.Mock(return_value=fake_res)

    @mock.patch.object(grouputils, 'get_rsrc_id')
    def test_get_attribute(self, mock_get_rsrc_id):
        stack = utils.parse_stack(template)
        mock_get_rsrc_id.side_effect = ['0', '1']
        rsrc = stack['group1']
        rsrc.get_output = mock.Mock(side_effect=exception.NotFound)
        self.assertEqual(['0', '1'], rsrc.FnGetAtt(rsrc.REFS))