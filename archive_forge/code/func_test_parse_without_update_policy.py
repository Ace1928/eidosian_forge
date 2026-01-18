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
def test_parse_without_update_policy(self):
    stack = utils.parse_stack(template)
    stack.validate()
    grp = stack['group1']
    self.assertFalse(grp.update_policy['rolling_update'])