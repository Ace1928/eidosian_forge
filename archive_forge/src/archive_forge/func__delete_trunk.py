import copy
from oslo_log import log as logging
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import trunk
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.tests import common
from heat.tests import utils
from neutronclient.common import exceptions as ncex
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
def _delete_trunk(self, stack):
    trunk = stack['trunk']
    scheduler.TaskRunner(trunk.delete)()
    self.assertEqual((trunk.DELETE, trunk.COMPLETE), trunk.state)