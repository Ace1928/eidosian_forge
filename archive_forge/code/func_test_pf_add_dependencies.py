import copy
from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from openstack import exceptions
from oslo_utils import excutils
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils
from heat.engine.clients.os import neutron
from heat.engine.hot import functions as hot_funcs
from heat.engine import node_data
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests import utils
def test_pf_add_dependencies(self):
    port = self.stack['port_floating']
    r_int = self.stack['router_interface']
    pf_port = self.stack['port_forwarding']
    deps = mock.MagicMock()
    dep_list = []

    def iadd(obj):
        dep_list.append(obj[1])
    deps.__iadd__.side_effect = iadd
    deps.graph.return_value = {pf_port: [port]}
    pf_port.add_dependencies(deps)
    self.assertEqual([r_int], dep_list)