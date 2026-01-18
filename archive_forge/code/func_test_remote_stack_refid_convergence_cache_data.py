import collections
import json
from unittest import mock
from heatclient import exc
from heatclient.v1 import stacks
from keystoneauth1 import loading as ks_loading
from oslo_config import cfg
from heat.common import exception
from heat.common.i18n import _
from heat.common import policy
from heat.common import template_format
from heat.engine.clients.os import heat_plugin
from heat.engine import environment
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.openstack.heat import remote_stack
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import template
from heat.tests import common as tests_common
from heat.tests import utils
def test_remote_stack_refid_convergence_cache_data(self):
    t = template_format.parse(parent_stack_template)
    cache_data = {'remote_stack': node_data.NodeData.from_dict({'uuid': mock.ANY, 'id': mock.ANY, 'action': 'CREATE', 'status': 'COMPLETE', 'reference_id': 'convg_xyz'})}
    stack = utils.parse_stack(t, cache_data=cache_data)
    rsrc = stack.defn['remote_stack']
    self.assertEqual('convg_xyz', rsrc.FnGetRefId())