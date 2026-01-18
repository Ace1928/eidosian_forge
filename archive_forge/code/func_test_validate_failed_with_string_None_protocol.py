from unittest import mock
from neutronclient.common import exceptions
from neutronclient.v2_0 import client as neutronclient
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import firewall
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_validate_failed_with_string_None_protocol(self):
    snippet = template_format.parse(firewall_rule_template)
    stack = utils.parse_stack(snippet)
    rsrc = stack['firewall_rule']
    props = dict(rsrc.properties)
    props['protocol'] = 'None'
    rsrc.t = rsrc.t.freeze(properties=props)
    rsrc.reparse()
    self.assertRaises(exception.StackValidationFailed, rsrc.validate)