from unittest import mock
from neutronclient.common import exceptions
from neutronclient.v2_0 import client as neutronclient
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def _create_l2_gateway(self, hot, reply):
    self.mockclient.create_l2_gateway.return_value = reply
    self.mockclient.show_l2_gateway.return_value = reply
    template = template_format.parse(hot)
    self.stack = utils.parse_stack(template)
    scheduler.TaskRunner(self.stack.create)()
    self.l2gw_resource = self.stack['l2gw']