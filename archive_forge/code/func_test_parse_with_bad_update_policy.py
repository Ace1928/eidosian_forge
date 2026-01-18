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
def test_parse_with_bad_update_policy(self):
    tmpl = tmpl_with_bad_updt_policy()
    stack = utils.parse_stack(tmpl)
    error = self.assertRaises(exception.StackValidationFailed, stack.validate)
    self.assertIn('foo', str(error))