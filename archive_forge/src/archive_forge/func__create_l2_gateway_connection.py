from unittest import mock
from neutronclient.v2_0 import client as neutronclient
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def _create_l2_gateway_connection(self):
    self.mockclient.create_l2_gateway_connection.return_value = self.mock_create_reply
    self.mockclient.show_l2_gateway_connection.return_value = self.mock_create_reply
    orig_template = template_format.parse(self.test_template)
    self.stack = utils.parse_stack(orig_template)
    scheduler.TaskRunner(self.stack.create)()
    self.l2gwconn_resource = self.stack['l2gw_conn']