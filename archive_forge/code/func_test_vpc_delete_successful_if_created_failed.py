from unittest import mock
import uuid
from heat.common import exception
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.aws.ec2 import subnet as sn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_vpc_delete_successful_if_created_failed(self):
    self.mock_create_network_failed()
    t = template_format.parse(self.test_template)
    stack = self.parse_stack(t)
    scheduler.TaskRunner(stack.create)()
    self.assertEqual((stack.CREATE, stack.FAILED), stack.state)
    self.mockclient.create_network.assert_called_once_with({'network': {'name': self.vpc_name}})
    scheduler.TaskRunner(stack.delete)()
    self.mockclient.delete_network.assert_not_called()