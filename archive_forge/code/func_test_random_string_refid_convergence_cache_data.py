import re
from unittest import mock
from testtools import matchers
from heat.common import exception
from heat.common import template_format
from heat.engine import node_data
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_random_string_refid_convergence_cache_data(self):
    t = template_format.parse(self.template_random_string)
    cache_data = {'secret1': node_data.NodeData.from_dict({'uuid': mock.ANY, 'id': mock.ANY, 'action': 'CREATE', 'status': 'COMPLETE', 'reference_id': 'xyz'})}
    stack = utils.parse_stack(t, cache_data=cache_data)
    rsrc = stack.defn['secret1']
    self.assertEqual('xyz', rsrc.FnGetRefId())