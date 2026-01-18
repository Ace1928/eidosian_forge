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
def test_update_time_not_sufficient(self):
    current = copy.deepcopy(template)
    self.stack = utils.parse_stack(current)
    self.current_grp = self.stack['group1']
    self.stack.timeout_secs = mock.Mock(return_value=200)
    err = self.assertRaises(ValueError, self.current_grp._update_timeout, 3, 100)
    self.assertIn('The current update policy will result in stack update timeout.', str(err))