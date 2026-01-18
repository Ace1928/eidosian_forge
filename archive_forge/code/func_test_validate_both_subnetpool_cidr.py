import copy
from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import openstacksdk
from heat.engine.hot import functions as hot_funcs
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.openstack.neutron import subnet
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.tests import common
from heat.tests import utils
def test_validate_both_subnetpool_cidr(self):
    self.patchobject(neutronV20, 'find_resourceid_by_name_or_id', return_value='new_pool')
    t = template_format.parse(neutron_template)
    props = t['resources']['sub_net']['properties']
    props['subnetpool'] = 'new_pool'
    stack = utils.parse_stack(t)
    self.patchobject(stack['net'], 'FnGetRefId', return_value='fc68ea2c-b60b-4b4f-bd82-94ec81110766')
    rsrc = stack['sub_net']
    ex = self.assertRaises(exception.ResourcePropertyConflict, rsrc.validate)
    msg = 'Cannot define the following properties at the same time: subnetpool, cidr.'
    self.assertEqual(msg, str(ex))