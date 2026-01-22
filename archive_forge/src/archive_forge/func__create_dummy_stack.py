import copy
from unittest import mock
from heat.common import exception
from heat.engine import node_data
from heat.engine.resources.openstack.heat import resource_chain
from heat.engine import rsrc_defn
from heat.objects import service as service_objects
from heat.tests import common
from heat.tests import utils
def _create_dummy_stack(self, expect_count=2, expect_attrs=None):
    self.stack = utils.parse_stack(TEMPLATE)
    snip = self.stack.t.resource_definitions(self.stack)['test-chain']
    chain = resource_chain.ResourceChain('test', snip, self.stack)
    attrs = {}
    refids = {}
    if expect_attrs is None:
        expect_attrs = {}
    for index in range(expect_count):
        res = str(index)
        attrs[index] = expect_attrs.get(res, res)
        refids[index] = 'ID-%s' % res
    names = [str(name) for name in range(expect_count)]
    chain._resource_names = mock.Mock(return_value=names)
    self._stub_get_attr(chain, refids, attrs)
    return chain