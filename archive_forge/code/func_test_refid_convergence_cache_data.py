from unittest import mock
import swiftclient.client as sc
from heat.common import exception
from heat.common import template_format
from heat.engine import node_data
from heat.engine.resources.openstack.swift import container as swift_c
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_refid_convergence_cache_data(self):
    cache_data = {'SwiftContainer': node_data.NodeData.from_dict({'uuid': mock.ANY, 'id': mock.ANY, 'action': 'CREATE', 'status': 'COMPLETE', 'reference_id': 'xyz_convg'})}
    stack = utils.parse_stack(self.t, cache_data=cache_data)
    rsrc = stack.defn['SwiftContainer']
    self.assertEqual('xyz_convg', rsrc.FnGetRefId())