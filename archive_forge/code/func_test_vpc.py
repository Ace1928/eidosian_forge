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
def test_vpc(self):
    self.mock_create_network()
    stack = self.create_stack(self.test_template)
    vpc = stack['the_vpc']
    self.assertResourceState(vpc, 'aaaa')
    self.validate_mock_create_network()
    self.assertEqual(3, self.mockclient.show_network.call_count)
    scheduler.TaskRunner(vpc.delete)()
    self.mockclient.show_network.assert_called_with('aaaa')
    self.assertEqual(4, self.mockclient.show_network.call_count)
    self.assertEqual(2, self.mockclient.list_routers.call_count)
    self.mockclient.list_routers.assert_called_with(name=self.vpc_name)
    self.mockclient.delete_router.assert_called_once_with('bbbb')
    self.mockclient.delete_network.assert_called_once_with('aaaa')