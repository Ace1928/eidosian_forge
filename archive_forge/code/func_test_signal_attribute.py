from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine import node_data
from heat.engine import resource
from heat.engine import scheduler
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests import utils
def test_signal_attribute(self):
    heat_plugin = self.stack.clients.client_plugin('heat')
    heat_plugin.get_heat_url = mock.Mock(return_value='http://server.test:8000/v1/')
    self.assertEqual('http://server.test:8000/v1/test_tenant_id/stacks/%s/%s/resources/my-policy/signal' % (self.stack.name, self.stack.id), self.policy.FnGetAtt('signal_url'))