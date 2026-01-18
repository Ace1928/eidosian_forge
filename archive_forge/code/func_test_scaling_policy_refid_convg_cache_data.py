from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine import node_data
from heat.engine import resource
from heat.engine import scheduler
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests import utils
def test_scaling_policy_refid_convg_cache_data(self):
    t = template_format.parse(as_template)
    cache_data = {'my-policy': node_data.NodeData.from_dict({'uuid': mock.ANY, 'id': mock.ANY, 'action': 'CREATE', 'status': 'COMPLETE', 'reference_id': 'convg_xyz'})}
    stack = utils.parse_stack(t, cache_data=cache_data)
    rsrc = stack.defn['my-policy']
    self.assertEqual('convg_xyz', rsrc.FnGetRefId())